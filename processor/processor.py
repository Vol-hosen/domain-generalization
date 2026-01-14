import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torch.nn.functional as F
from model import objectives


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "cons_loss": AverageMeter(),
        "delta_sim": AverageMeter(),
        "top1_consistency": AverageMeter(),
        "mixup_loss": AverageMeter(),
        "topo_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # --- 定义一个内部函数，用于复用 Loss 计算逻辑 ---
            def calculate_loss(batch_data):
                ret = model(batch_data)
                # 初始化 cons_loss 相关变量
                cons_loss = torch.tensor(0.0).to(device)
                delta_sim = torch.tensor(0.0).to(device)
                top1_consistency = torch.tensor(0.0).to(device)
                mixup_loss = torch.tensor(0.0).to(device)
                topo_loss = torch.tensor(0.0).to(device)

                if epoch > args.cons_warmup_epochs:
                    i_feats = ret['i_feats']
                    t_feats = ret['t_feats']


                    if args.use_mixup and args.mixup_alpha > 0:

                        mixed_i, mix_idx, lam, mix_mask =objectives.manifold_mixup_nn_threshold(
                            i_feats,
                            alpha=args.mixup_alpha,
                            sim_th=args.mixup_sim_th,     # e.g. 0.5
                            detach=args.mixup_detach
                        )

                        # normalize
                        mixed_i = F.normalize(mixed_i, dim=1)
                        t_feats_norm = F.normalize(t_feats, dim=1)

                        mix_temp = getattr(args, "mixup_temp", 0.1)
                        logits_mix = (mixed_i @ t_feats_norm.t()) / mix_temp

                        # ---- build soft labels only for mixed samples ----
                        with torch.no_grad():
                            B = logits_mix.size(0)
                            labels_mix = torch.zeros_like(logits_mix)

                            for i in range(B):
                                if not mix_mask[i]:
                                    # no mix: standard one-hot
                                    labels_mix[i, i] = 1.0
                                else:
                                    j = mix_idx[i]
                                    labels_mix[i, i] = lam
                                    labels_mix[i, j] = 1 - lam

                        log_probs = F.log_softmax(logits_mix, dim=1)
                        mixup_loss = -(labels_mix * log_probs).sum(dim=1)

                        # ---- only average over mixed samples (更干净) ----
                        if mix_mask.any():
                            mixup_loss = mixup_loss[mix_mask].sum() / B
                        else:
                            mixup_loss = torch.tensor(0.0, device=i_feats.device)

                    # Phase-1: random embedding perturbation
                    epsilon = args.cons_eps 
                    with torch.no_grad():
                        #noise = torch.randn_like(i_feats)
                        #noise = F.normalize(noise, dim=1) 
                        # 建议比例 0.1~0.2
                        mask = (torch.rand_like(i_feats) > 0.15).float()
                        

                        
                    i_feats_pert = F.normalize(i_feats * mask, dim=1)
                    #i_feats_pert = F.normalize(i_feats + epsilon * noise, dim=1)
                    i_feats = F.normalize(i_feats, dim=1)
                    t_feats = F.normalize(t_feats, dim=1)

                    # Consistency Loss (KL)
                    temp = 0.07
                    logits_orig = i_feats @ t_feats.t() / temp
                    logits_pert = i_feats_pert @ t_feats.t() / temp

                    K = 5 
                    topk_indices = logits_orig.topk(K, dim=1).indices
                    logits_orig_k = torch.gather(logits_orig, 1, topk_indices)
                    logits_pert_k = torch.gather(logits_pert, 1, topk_indices)

                    p_orig_k = F.softmax(logits_orig_k.detach(), dim=-1)
                    log_p_pert_k = F.log_softmax(logits_pert_k, dim=-1)
                    cons_loss = F.kl_div(log_p_pert_k, p_orig_k.detach(), reduction='batchmean')

                    # Logging 辅助计算
                    with torch.no_grad():
                        sim_orig = F.cosine_similarity(i_feats, t_feats, dim=1)
                        sim_pert = F.cosine_similarity(i_feats_pert, t_feats, dim=1)
                        delta_sim = torch.abs(sim_orig - sim_pert).mean()
                        top1_orig = logits_orig.argmax(dim=1)
                        top1_pert = logits_pert.argmax(dim=1)
                        top1_consistency = (top1_orig == top1_pert).float().mean()

                    #topo-loss
                    if 'topo' in args.loss_names:
                        wi = objectives.compute_pair_reliability(
                            ret['i_feats'], ret['t_feats'], args.temperature)
                        topo_loss = objectives.compute_topo_loss(
                            ret['i_feats'], ret['t_feats'], args.temperature,wi)
                # 计算总 Loss
                current_total_loss = sum([v for k, v in ret.items() if "loss" in k])
                current_total_loss += args.cons_loss_weight * cons_loss
                current_total_loss += args.mixup_loss_weight * mixup_loss
                current_total_loss += args.topo_loss_weight * topo_loss
                
                return current_total_loss, ret, cons_loss, delta_sim, top1_consistency,mixup_loss,topo_loss

            # --- 真正的训练步开始 ---
            #TODO:SAM实现有bug，先不启用，理论上启用后效果会更好
            if args.use_sam:
                # SAM 第一次迭代：正常计算梯度并寻找“最坏”点
                total_loss, ret, cons_l, d_sim, t1_cons, mixup_loss, topo_loss = calculate_loss(batch)
                total_loss.backward()
                optimizer.first_step(zero_grad=True)

                # SAM 第二次迭代：在最坏点处计算梯度
                # 注意：必须重新运行 calculate_loss，因为参数已经变了
                total_loss_2, _, _, _, _, _,_ = calculate_loss(batch)
                total_loss_2.backward()
                optimizer.second_step(zero_grad=True)
            else:
                # 常规训练
                optimizer.zero_grad()
                total_loss, ret, cons_l, d_sim, t1_cons, mixup_loss,topo_loss = calculate_loss(batch)
                total_loss.backward()
                optimizer.step()

            synchronize()
            

            # --- Meters 更新 (使用第一次 forward 的结果即可) ---
            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['cons_loss'].update(cons_l.item(), batch_size)
            meters['mixup_loss'].update(mixup_loss.item(), batch_size)
            meters['topo_loss'].update(topo_loss.item(), batch_size)
            meters['delta_sim'].update(d_sim.item(), batch_size)
            meters['top1_consistency'].update(t1_cons.item(), batch_size)

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        #info_str += f", {k}: {v.avg:.4f}"
                        if k == 'delta_sim' or k == 'cons_loss' or k == 'topo_loss':
                            # 对 delta_sim 使用科学计数法，保留 2 位小数
                            info_str += f", {k}: {v.avg:.2e}" 
                        else:
                            # 其他指标依然使用原来的 4 位小数
                            info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
