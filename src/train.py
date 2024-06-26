import torch
import torch.nn as nn
from tqdm import tqdm
from src.models import AggregatedLoss


def train_clip(film_encoder, text_encoder, train_loader, val_loader,
               epochs, lr, device='cpu', iter_verbose=500, folder=None, writer=None):
    aggregated_loss = AggregatedLoss(device=device).to(device)
    optimizer = torch.optim.Adam(list(film_encoder.parameters()) + list(text_encoder.parameters()), lr=lr)

    global_batch = 0

    for epoch in range(epochs):
        film_encoder.train()
        text_encoder.train()

        film_encoder.to(device)
        text_encoder.to(device)
        running_loss = 0.0
        total_batches = 0
        running_classification_loss = 0
        running_contrastive_loss = 0
        total_train_correct = 0
        total_train_samples = 0
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            film_ids = batch['film_ids'].to(device)
            target_id = batch['target_id'].to(device)
            descriptions = batch['descriptions'].to(device)
            attention_masks = batch['attention_masks'].to(device)

            film_logits, film_embeddings, _ = film_encoder(film_ids)
            text_embeddings = text_encoder(descriptions, attention_masks)

            loss = aggregated_loss(film_ids, film_logits, target_id, film_embeddings, text_embeddings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_classification_loss += aggregated_loss.classification_loss.item()
            running_contrastive_loss += aggregated_loss.contrastive_loss.item()
            _, predicted = torch.max(film_logits, 1)
            total_train_correct += (predicted == target_id).sum().item()
            total_train_samples += target_id.size(0)
            total_batches += 1
            global_batch += 1

            if (i + 1) % iter_verbose == 0 or i == len(train_loader) - 1:
                train_accuracy = total_train_correct / total_train_samples
                print(f"Epoch {epoch + 1}, Batch {i + 1}")
                print(f"Accuracy: {train_accuracy:.4f}")
                print(f"Agreggated loss: {running_loss / total_batches:.4f}")
                print(f"Classification loss: {running_classification_loss / total_batches:.4f}")
                print(f"Contrastive loss: {running_contrastive_loss / total_batches:.4f}")
                print()
                if writer is not None:
                    writer.add_scalar("Accuracy/train", train_accuracy, global_batch)
                    writer.add_scalar("Agreggated loss/train", running_loss / total_batches, global_batch)
                    writer.add_scalar("Classification loss/train", running_classification_loss / total_batches, global_batch)
                    writer.add_scalar("Contrastive loss/train", running_contrastive_loss / total_batches, global_batch)
                total_train_correct = 0
                total_train_samples = 0

        if (epoch + 1) % 1 == 0:
            film_encoder.eval()
            text_encoder.eval()

            total_loss = 0
            total_correct = 0
            total_samples = 0
            total_classification_loss = 0
            total_contrastive_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    film_ids = batch['film_ids'].to(device)
                    target_id = batch['target_id'].to(device)
                    descriptions = batch['descriptions'].to(device)
                    attention_masks = batch['attention_masks'].to(device)

                    film_logits, film_embeddings, _ = film_encoder(film_ids)
                    text_embeddings = text_encoder(descriptions, attention_masks)

                    loss = aggregated_loss(film_ids, film_logits, target_id, film_embeddings, text_embeddings,
                                           train=False)

                    total_loss += loss.item()
                    total_classification_loss += aggregated_loss.classification_loss.item()
                    total_contrastive_loss += aggregated_loss.contrastive_loss.item()
                    _, predicted = torch.max(film_logits, 1)
                    total_correct += (predicted == target_id).sum().item()
                    total_samples += target_id.size(0)

            average_loss = total_loss / len(val_loader)
            accuracy = total_correct / total_samples
            print(f"Epoch {epoch + 1}: Val Loss: {average_loss:.4f}, Val Accuracy: {accuracy:.4f}")
            print(f"Val Classification loss: {total_classification_loss / len(val_loader):.4f}")
            print(f"Val Contrastive loss: {total_contrastive_loss / len(val_loader):.4f}")
            if writer is not None:
                writer.add_scalar("Accuracy/validation", accuracy, (epoch + 1))
                writer.add_scalar("Classification loss/validation", total_classification_loss / len(val_loader), (epoch + 1))
                writer.add_scalar("Contrastive loss/validation", total_contrastive_loss / len(val_loader), (epoch + 1))

        if folder:
            torch.save(film_encoder.state_dict(), f'{folder}/film_encoder_weights_{epoch}.pth')
            torch.save(text_encoder.state_dict(), f'{folder}/text_encoder_weights_{epoch}.pth')


def train_recommender(model, train_loader, val_loader, epochs, lr, device='cpu'):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        print()

        model.train()
        model.to(device)
        train_loss = 0.0
        train_correct = 0
        train_samples = 0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            film_logits, film_embeddings, _ = model(inputs)
            loss = criterion(film_logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(film_logits, 1)
            train_correct += (predicted == targets).sum().item()
            train_samples += targets.size(0)

        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Train Accuracy: {train_correct / train_samples:.4f}")
        print()

        if (epoch + 1) % 1 == 0:
            model.eval()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    film_logits, film_embeddings, _ = model(inputs)
                    loss = criterion(film_logits, targets)
                    total_loss += loss.item()
                    _, predicted = torch.max(film_logits, 1)
                    total_correct += (predicted == targets).sum().item()
                    total_samples += targets.size(0)

            average_loss = total_loss / len(val_loader)
            accuracy = total_correct / total_samples
            print(f"Val Loss: {average_loss:.4f}")
            print(f"Val Accuracy: {accuracy:.4f}")
            print()
