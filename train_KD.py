from utils import data_setup, engine, save
from architecture import Mobilenet, Resnet
from argparse import ArgumentParser
import torch
from torchinfo import summary
from torchvision import datasets, transforms
if __name__ == "__main__":
    parser = ArgumentParser(description='Train classification')
    parser.add_argument('--work-dir', default='models', help='the dir to save logs and models')
    parser.add_argument("--train_dir", default='data/train', type=str)
    parser.add_argument("--test_dir", default='data/test', type=str)
    parser.add_argument('--architecture', default='KD', type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--img_size", default=112, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    args = parser.parse_args()

    print(f'Training {args.architecture} model with hyper-params:')
    devices = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = args.train_dir
    test_dir = args.test_dir

    IMG_SIZE = args.img_size
    MEANS = (0.46295794, 0.46194877, 0.4847407)
    STDS = (0.19444681, 0.19439201, 0.19383532)
    # Create transform pipeline manually
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS),])
    
    # Load train dataset 
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                transform=manual_transforms, # use manually created transforms
                                                                                batch_size=args.batch_size,
                                                                                num_workers=args.num_workers)

    model_teacher = Resnet.ResNet50(num_classes=len(class_names))
    model_student = Mobilenet.MobileNetV3(config_name = 'small', num_classes=len(class_names))
    
    total_params_deep = "{:,}".format(sum(p.numel() for p in model_teacher.parameters()))
    print(f"Resnet parameters (Teacher): {total_params_deep}")
    total_params_light = "{:,}".format(sum(p.numel() for p in model_student.parameters()))
    print(f"MobileNetV3 parameters (Student): {total_params_light}")

    # Training model Teacher
    # Setup the loss function and optimizer for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_teacher.parameters(), lr=args.lr, momentum=0.9)

    # Set the seeds
    engine.set_seeds()

    print(f"Training model Teacher:")
    # Train the model and save the training results to a dictionary
    results_teacher = engine.train(model=model_teacher,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    epochs=args.num_epochs,
                                    work_dir=args.work_dir,
                                    architecture=args.architecture,
                                    device=devices)

    _, test_accuracy_teacher = engine.test_step(model=model_teacher,
                                                    dataloader=test_dataloader,
                                                    loss_fn=loss_fn,
                                                    device=devices)
    # Training model student
    # Set the seeds
    engine.set_seeds()
    # Setup the loss function and optimizer for multi-class classification
    loss_fn_st = torch.nn.CrossEntropyLoss()
    optimizer_st  = torch.optim.SGD(model_student.parameters(), lr=args.lr, momentum=0.9)
    print(f"Training model Student:")
    results_student = engine.train(model=model_student,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    optimizer=optimizer_st,
                                    loss_fn=loss_fn_st,
                                    epochs=args.num_epochs,
                                    work_dir=args.work_dir,
                                    architecture=args.architecture,
                                    device=devices)

    _, test_accuracy_student = engine.test_step(model=model_student,
                                                dataloader=test_dataloader,
                                                loss_fn=loss_fn_st,
                                                device=devices)

    print(f"Training Knowledge Distillation:")
    new_model_student = Mobilenet.MobileNetV3(config_name = 'small', num_classes=len(class_names))
    results_kd = engine.train_knowledge_distillation(teacher=model_teacher, 
                                                    student=new_model_student, 
                                                    train_loader=train_dataloader, 
                                                    epochs=args.num_epochs, 
                                                    learning_rate=args.lr, 
                                                    T=2, 
                                                    soft_target_loss_weight=0.25, 
                                                    ce_loss_weight=0.75, 
                                                    device=devices)

    _, test_accuracy_light_ce_and_kd = engine.test_step(model=new_model_student,
                                                    dataloader=test_dataloader,
                                                    loss_fn=loss_fn_st,
                                                    device=devices)


    print(f"Teacher accuracy: {test_accuracy_teacher:.2f}%")
    print(f"Student accuracy without teacher: {test_accuracy_student:.2f}%")
    print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.4f}")