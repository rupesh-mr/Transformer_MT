import torch
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

def compute_rdrop_loss(model, src, tgt_input, tgt_output, src_mask, tgt_mask, alpha=5.0, ignore_index=0):
    # Forward pass twice with different dropout masks
    logits_1 = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)  # (B, T, V)
    logits_2 = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

    # Flatten for CE loss
    # B, T, V = logits_1.size()
    # logits_1_flat = logits_1.reshape(B * T, V)
    # logits_2_flat = logits_2.reshape(B * T, V)
    # tgt_output_flat = tgt_output.reshape(B * T)
    
    B, T, V = logits_1.size()       # changes 1 vikas has done as per the previous file
    logits_1_flat = logits_1.view(B * T, V)
    logits_2_flat = logits_2.view(B * T, V)
    #tgt_output_flat = tgt_output.view(B * T)
    tgt_output_flat = tgt_output.reshape(-1)

    # Cross-entropy loss for both outputs
    ce_loss_1 = F.cross_entropy(logits_1_flat, tgt_output_flat, ignore_index=ignore_index)
    ce_loss_2 = F.cross_entropy(logits_2_flat, tgt_output_flat, ignore_index=ignore_index)
    ce_loss = (ce_loss_1 + ce_loss_2) / 2

    # Compute KL divergence only on non-pad positions
    log_probs_1 = F.log_softmax(logits_1, dim=-1)  # (B, T, V)
    log_probs_2 = F.log_softmax(logits_2, dim=-1)
    #probs_1 = F.softmax(logits_1, dim=-1)
    #probs_2 = F.softmax(logits_2, dim=-1)
    
    

    # Per-token KL divergence (B, T)
    
    kl_1 = F.kl_div(log_probs_1, log_probs_2, reduction='none', log_target=True).sum(-1)
    kl_2 = F.kl_div(log_probs_2, log_probs_1, reduction='none', log_target=True).sum(-1)
    #kl_1 = F.kl_div(log_probs_1, probs_2, reduction='none').sum(-1)  # (B, T)
    #kl_2 = F.kl_div(log_probs_2, probs_1, reduction='none').sum(-1)

    # Mask out padding
    non_pad_mask = (tgt_output != ignore_index).float()  # (B, T)
    kl = ((kl_1 + kl_2) / 2) * non_pad_mask  # (B, T)

    kl_div = kl.sum() / non_pad_mask.sum()  # mean over non-pad tokens

    total_loss = ce_loss + alpha * kl_div
    return total_loss

def train_model(model, dataloader,valid_dataloader, optimizer,scheduler, criterion, device, epochs=5,start_epoch=10, checkpoint_path=None):
    total_batches = len(dataloader)
    for epoch in range(1+start_epoch, start_epoch+epochs + 1):
        model.train()
        print(f"Epoch {epoch}/{start_epoch+epochs}")
        with open('training_log.txt', 'a') as log:
                log.write(f"Epoch {epoch}/{start_epoch+epochs}\n")
        epoch_start_time = time.time()
        batch_times = []
        total_loss=0
        for i, batch in enumerate(dataloader, 1):
            batch_start = time.time()

            src, tgt_input, tgt_output, src_mask, tgt_mask = batch

            src        = src.to(device)
            tgt_input  = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            src_mask   = src_mask.to(device)
            tgt_mask   = tgt_mask.to(device)

            #logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            #B, Tm1, V = logits.size()
            # assert tgt_output.min() >= 0 and tgt_output.max() < V, (
            #     f"Target out of range: got [{tgt_output.min()}, {tgt_output.max()}], "
            #     f"but vocab size is {V}"
            # )


            # with torch.no_grad():
            #     lmin, lmax = logits.min().item(), logits.max().item()
            #     assert torch.isfinite(logits).all(), f"Non-finite logits detected: [{lmin}, {lmax}]"
            # Compute loss with R-Drop
            loss = compute_rdrop_loss(model, src, tgt_input,tgt_output, src_mask, tgt_mask, alpha=5.0)
            # Backprop
            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}") #here i put according to previous
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)

            avg_batch_time = sum(batch_times) / len(batch_times)
            batches_left = total_batches - i
            est_time_left = batches_left * avg_batch_time
            total_loss += loss.item()
            if i%10==0:
              with open('training_log.txt', 'a') as log:
                log.write(f"Batch {i}/{total_batches} - Loss: {loss.item():.4f}\n")
              print(f"Batch {i}/{total_batches} - Loss: {loss.item():.4f} - "
                    f"Batch time: {batch_time:.2f}s - Est. time left: {est_time_left:.2f}s")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"\nEpoch {epoch+start_epoch} completed in {epoch_duration:.2f}s - Last Loss: {loss.item():.4f} - Average Loss: {total_loss / total_batches:.4f}")
        with open('training_log.txt', 'a') as log:
            log.write(f"Epoch {epoch} completed in {epoch_duration:.2f}s - Last Loss: {loss.item():.4f} - Average Loss: {total_loss / total_batches:.4f}\n\n")

        if checkpoint_path:
            import os
            dir=os.path.dirname(checkpoint_path+"")
            if dir:
             os.makedirs(dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item()
            }, checkpoint_path+'_'+str(epoch)+'.pth')
            print(f"Checkpoint saved to {checkpoint_path+'_'+str(epoch)+'.pth'}")

        #if checkpoint_path:
            #import os
            #dir=os.path.dirname(checkpoint_path+"")
            #if dir:
             #os.makedirs(dir, exist_ok=True)
            #torch.save({
               # 'epoch': epoch,
               # 'model_state_dict': model.state_dict(),
              #  'optimizer_state_dict': optimizer.state_dict(),
              #  'scheduler_state_dict': scheduler.state_dict(),
             #   'loss': loss.item()
            #}, checkpoint_path + '_' + str(epoch) +' .pth')
          # print(f"Checkpoint saved to {checkpoint_path + '_' + str(epoch)+ '.pth'}")
        # Validation
        model.eval()
        total_valid_loss = 0

        
        with torch.inference_mode():
            for i, batch in enumerate(valid_dataloader, 1):
                src, tgt_input, tgt_output, src_mask, tgt_mask = batch

                src        = src.to(device)
                tgt_input  = tgt_input.to(device)
                tgt_output = tgt_output.to(device)
                src_mask   = src_mask.to(device)
                tgt_mask   = tgt_mask.to(device)

                logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))   # logits.size(-1) as logits.size()
                total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        print(f"Validation Loss after Epoch {epoch}: {avg_valid_loss:.4f}") #this uncomment
        with open('validation_log.txt', 'a') as log:
            log.write(f"Validation Loss after Epoch {epoch}: {avg_valid_loss:.4f}\n")
            # log.write(f"Epoch {epoch}/{epochs}, Valid Loss: {avg_valid_loss:.4f}\n") #this extra chnages
            
        print(f"Epoch {epoch}/{epochs}, Valid Loss: {avg_valid_loss:.4f}") # extra put
# Function to load checkpoint
def load_checkpoint(model, optimizer,scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}, last loss: {checkpoint['loss']}")

def get_inverse_sqrt_warmup_scheduler(optimizer, warmup_steps, peak_lr, initial_lr=1e-5):
    def lr_lambda(step):
        if step == 0:
            return initial_lr / peak_lr
        if step < warmup_steps:
            # Linear interpolate multiplier from initial_lr/peak_lr to 1
            return ((peak_lr - initial_lr) * step / warmup_steps + initial_lr) / peak_lr
        else:
            return (warmup_steps ** 0.5) * (step ** -0.5)

    for param_group in optimizer.param_groups:
        param_group['lr'] = peak_lr

    return LambdaLR(optimizer, lr_lambda)
