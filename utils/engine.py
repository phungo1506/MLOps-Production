import torch 
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from . import save

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          work_dir: str,
          architecture: str,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    best_metric = float('-inf')  # Initialize the best metric value

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )
        if test_acc > best_metric:
            best_metric = test_acc
            # Save the model's state
            save.save_model(model, work_dir, architecture)
            print(f'Saved best model with validation accuracy: {best_metric:.4f}')
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

   
def train_knowledge_distillation(teacher: torch.nn.Module, 
                                 student: torch.nn.Module, 
                                 train_loader: torch.utils.data.DataLoader,
                                 test_dataloader: torch.utils.data.DataLoader,  
                                 epochs: int, 
                                 learning_rate: float, 
                                 T: float, 
                                 soft_target_loss_weight: float, 
                                 ce_loss_weight: float,
                                 work_dir: str,
                                 architecture: str,  
                                 device: torch.device):
    """
    Train a student model using knowledge distillation from a teacher model.

    Args:
        teacher (torch.nn.Module): Pre-trained teacher model.
        student (torch.nn.Module): Student model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for optimizer.
        T (float): Temperature for soft targets.
        soft_target_loss_weight (float): Weight for the soft target loss.
        ce_loss_weight (float): Weight for the cross-entropy loss with true labels.
        device (torch.device): Device to perform computations on.

    Returns:
        None
    """
    # Set the teacher model to evaluation mode and send to device
    best_metric = float('-inf')  # Initialize the best metric value
    teacher.eval()
    teacher.to(device)

    # Set the student model to training mode and send to device
    student.train()
    student.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, eps=1e-08, weight_decay=0.01)

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        correct = 0
        total = 0

        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass for teacher
            with torch.no_grad():
                teacher_outputs = teacher(images)

            # Forward pass for student
            student_outputs = student(images)

            # Compute the distillation loss (soft target loss)
            soft_targets = torch.softmax(teacher_outputs / T, dim=1)
            student_logits = student_outputs / T
            distillation_loss = torch.nn.KLDivLoss()(torch.log_softmax(student_logits, dim=1), soft_targets) * (T * T)

            # Compute the cross-entropy loss (hard target loss)
            ce_loss = criterion(student_outputs, labels)

            # Combine the two losses
            loss = soft_target_loss_weight * distillation_loss + ce_loss_weight * ce_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(student_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        test_loss, test_acc = test_step(model=student,
                                        dataloader=test_dataloader,
                                        loss_fn=criterion,
                                        device=device)
        print(f'Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f} | Test Loss: {test_loss:.4f}  | Test Accuracy: {test_acc:.4f}')
        
        if test_acc > best_metric:
          best_metric = test_acc
          # Save the model's state
          save.save_model(student, work_dir, architecture)
          print(f'Saved best model with validation accuracy: {best_metric:.4f}')
        

    print('Training complete')