import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torch.nn.functional as F


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
        "top1_consistency": AverageMeter()
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

            ret = model(batch)
            if epoch > args.cons_warmup_epochs:
                

                i_feats = ret['i_feats']
                t_feats = ret['t_feats']

                # Phase-1: random embedding perturbation
                epsilon = args.cons_eps  # e.g. 1e-3
                with torch.no_grad():
                    # 1. 生成原始噪声
                    noise = torch.randn_like(i_feats)
                    # 2. 投影到切平面: noise = noise - (noise · feat) * feat
                    # 这样确保 noise ⊥ i_feats
                    #noise = noise - (noise * i_feats).sum(dim=1, keepdim=True) * i_feats
                    # 3. 单位化噪声方向，并缩放到 epsilon 步长
                    noise = F.normalize(noise, dim=1) 

                # 4. 在切平面方向施加扰动并重新归一化（回到球面上）
                i_feats_pert = F.normalize(i_feats + epsilon * noise, dim=1)
                i_feats = F.normalize(i_feats, dim=1)
                t_feats = F.normalize(t_feats, dim=1)


                # --- Consistency Loss ---
                temp = 0.07
                logits_orig = i_feats @ t_feats.t() / temp
                logits_pert = i_feats_pert @ t_feats.t() / temp

                # 3. 将 MSE 换成 KL 散度
                p_orig = F.softmax(logits_orig.detach(), dim=-1)
                log_p_pert = F.log_softmax(logits_pert, dim=-1)

                K = 5 
                # 1. 找到原始输出中相似度最高的 K 个索引
                topk_indices = logits_orig.topk(K, dim=1).indices # [Batch, K]

                # 2. 从原始 Logits 中提取这 K 个位置的原始得分 (注意：是取 logits 而不是 p)
                logits_orig_k = torch.gather(logits_orig, 1, topk_indices)
                logits_pert_k = torch.gather(logits_pert, 1, topk_indices)

                # 3. 对这 K 个得分重新进行 Softmax，构建“局部分布”
                # 这样确保了这 K 个概率加起来等于 1
                p_orig_k = F.softmax(logits_orig_k.detach(), dim=-1)
                log_p_pert_k = F.log_softmax(logits_pert_k, dim=-1)

                # 4. 计算局部 KL 散度
                cons_loss = F.kl_div(log_p_pert_k, p_orig_k.detach(), reduction='batchmean')

                # 计算两个分布的一致性
                #cons_loss = F.kl_div(log_p_pert, p_orig.detach(), reduction='batchmean')
                
                sim_orig = F.cosine_similarity(i_feats, t_feats, dim=1)
                sim_pert = F.cosine_similarity(i_feats_pert, t_feats, dim=1)

                

                # --- Logging (非常重要) ---
                with torch.no_grad():
                    delta_sim = torch.abs(sim_orig - sim_pert).mean()
                    # 可以通过 wandb 或 tensorboard 监控这个值
                    top1_orig = logits_orig.argmax(dim=1)
                    top1_pert = logits_pert.argmax(dim=1)
                    top1_consistency = (top1_orig == top1_pert).float().mean()
            else:
                cons_loss = torch.tensor(0.0).to(device)
                delta_sim = torch.tensor(0.0).to(device)
                top1_consistency = torch.tensor(0.0).to(device)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            total_loss += args.cons_loss_weight * cons_loss

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['cons_loss'].update(cons_loss.item(), batch_size)
            meters['delta_sim'].update(delta_sim.item(), batch_size)
            meters['top1_consistency'].update(top1_consistency.item(), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        #info_str += f", {k}: {v.avg:.4f}"
                        if k == 'delta_sim' or k == 'cons_loss':
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
