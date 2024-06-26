from utils import data_setup, engine, engine_quant, save
from architecture import Mobilenet, Resnet
from argparse import ArgumentParser
import torch
from torchvision import datasets, transforms
from torchinfo import summary

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = args.train_dir
    test_dir = args.test_dir

    IMG_SIZE = args.img_size
    MEANS = (0.46295794, 0.46194877, 0.4847407)
    STDS = (0.19444681, 0.19439201, 0.19383532)
    # Create transform pipeline manually
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS),
    ])

    # Load train dataset
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                   test_dir=test_dir,
                                                                                   transform=manual_transforms,  # use manually created transforms
                                                                                   batch_size=args.batch_size,
                                                                                   num_workers=args.num_workers)

    # model_teacher = Resnet.ResNet101(num_classes=len(class_names)).to(device)
    model_teacher = Mobilenet.MobileNetV3(config_name='large', num_classes=len(class_names)).to(device)
    model_student = Mobilenet.MobileNetV3(config_name='small_KD', num_classes=len(class_names)).to(device)

    total_params_deep = "{:,}".format(sum(p.numel() for p in model_teacher.parameters()))
    # print(f"Resnet parameters (Teacher): {total_params_deep}")
    print(f"MobileNetV3 Large parameters (Teacher): {total_params_deep}")
    total_params_light = "{:,}".format(sum(p.numel() for p in model_student.parameters()))
    print(f"MobileNetV3 parameters (Student): {total_params_light}")


    # Print a summary of our custom model using torchinfo (uncomment for actual output)
    print("="*50)
    print("Architecture model Teacher:")
    summary(model=model_teacher,
            input_size=(128, 3,  IMG_SIZE, IMG_SIZE), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    print("="*50)
    print("Architecture model Student:")
    summary(model=model_student,
            input_size=(128, 3,  IMG_SIZE, IMG_SIZE), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])


    # Training model Teacher
    # Setup the loss function and optimizer for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_teacher.parameters(), lr=args.lr, eps=1e-08, weight_decay=0.01)

    # Set the seeds
    engine.set_seeds()

    print("="*50)
    print(f"Training model Teacher:")
    # Train the model and save the training results to a dictionary
    results_teacher = engine.train(model=model_teacher,
                                   train_dataloader=train_dataloader,
                                   test_dataloader=test_dataloader,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   epochs=args.num_epochs,
                                   work_dir=args.work_dir,
                                   architecture=args.architecture + '_teacher',
                                   device=device)

    _, test_accuracy_teacher, _ = engine.test_step(model=model_teacher,
                                                dataloader=test_dataloader,
                                                loss_fn=loss_fn,
                                                device=device)

    # Training model student
    # Set the seeds
    engine.set_seeds()
    # Setup the loss function and optimizer for multi-class classification
    loss_fn_st = torch.nn.CrossEntropyLoss()
    optimizer_st = torch.optim.AdamW(model_student.parameters(), lr=args.lr, eps=1e-08, weight_decay=0.01)
    print("="*50)
    print(f"Training model Student:")
    results_student = engine.train(model=model_student,
                                   train_dataloader=train_dataloader,
                                   test_dataloader=test_dataloader,
                                   optimizer=optimizer_st,
                                   loss_fn=loss_fn_st,
                                   epochs=args.num_epochs,
                                   work_dir=args.work_dir,
                                   architecture=args.architecture + '_student',
                                   device=device)

    best_model_student = Mobilenet.MobileNetV3(config_name='small_KD', num_classes=len(class_names)).to(device)
    path_student_model = 'models/' + args.architecture + '_student_best.pth'
    best_model_student.load_state_dict(torch.load(path_student_model))
    _, test_accuracy_student, _ = engine.test_step(model=best_model_student,
                                                dataloader=test_dataloader,
                                                loss_fn=loss_fn_st,
                                                device=device)

    print("="*50)
    print(f"Training Knowledge Distillation:")
    best_model_teacher = Mobilenet.MobileNetV3(config_name='large', num_classes=len(class_names)).to(device)
    # best_model_teacher = Resnet.ResNet101(num_classes=len(class_names)).to(device)
    path_teacher_model = 'models/' + args.architecture + '_teacher_best.pth'
    print(path_teacher_model)
    best_model_teacher.load_state_dict(torch.load(path_teacher_model))
    new_model_student = Mobilenet.MobileNetV3(config_name='small_KD', num_classes=len(class_names)).to(device)
    results_kd = engine.train_knowledge_distillation(teacher=best_model_teacher,
                                                     student=new_model_student,
                                                     train_loader=train_dataloader,
                                                     test_dataloader=test_dataloader,
                                                     epochs=args.num_epochs,
                                                     learning_rate=0.0001,
                                                     T=2,
                                                     soft_target_loss_weight=0.25,
                                                     ce_loss_weight=0.75,
                                                     work_dir=args.work_dir,
                                                     architecture=args.architecture,
                                                     device=device)

    best_model_KD = Mobilenet.MobileNetV3(config_name='small_KD', num_classes=len(class_names)).to(device)
    path_KD_model = 'models/' + args.architecture + '_best.pth'
    best_model_KD.load_state_dict(torch.load(path_KD_model))
    print(path_KD_model)
    _, test_accuracy_light_ce_and_kd, _ = engine.test_step(model=best_model_KD,
                                                        dataloader=test_dataloader,
                                                        loss_fn=loss_fn_st,
                                                        device=device)


    # print(f"Teacher accuracy: {test_accuracy_teacher:.4f} | Size model Teacher: {size_teacher}")
    print("="*20)
    print("Model Teacher Profile:")
    engine_quant.profile(best_model_teacher, test_dataloader, loss_fn, 'cpu')
    print("="*20)
    print("Model without Teacher Profile:")
    engine_quant.profile(best_model_student, test_dataloader, loss_fn, 'cpu')
    print("="*20)
    print("Model with CE + KD Profile:")
    engine_quant.profile(best_model_KD, test_dataloader, loss_fn, 'cpu')
    # print(f"Student accuracy without teacher: {test_accuracy_student:.4f}")
    # print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.4f} | Size model Student: {size_student}")
