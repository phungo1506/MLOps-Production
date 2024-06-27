from utils import data_setup, engine, engine_quant, save
from architecture import Mobilenet, Resnet
from argparse import ArgumentParser
import torch
from torchinfo import summary
from torchvision import datasets, transforms
from torch.quantization.quantize_fx import prepare_fx, convert_fx

if __name__ == "__main__":
    parser = ArgumentParser(description='Train classification')
    parser.add_argument('--work_dir', default='models', help='the dir to save logs and models')
    parser.add_argument("--train_dir", default='data/train', type=str)
    parser.add_argument("--test_dir", default='data/test', type=str)
    parser.add_argument('--architecture', default='MobileNetv3', type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    args = parser.parse_args()

    print(f'Training {args.architecture} model with hyper-params:')
    devices = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = args.train_dir
    test_dir = args.test_dir
    print(devices)
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
    if args.architecture == "MobileNetv3":
        model = Mobilenet.MobileNetV3(config_name='small', num_classes=len(class_names))
        model_best = Mobilenet.MobileNetV3(config_name='small', num_classes=len(class_names))
    if args.architecture == "Resnet":
        model = Resnet.ResNet50(num_classes=len(class_names))
        model_best = Resnet.ResNet50(num_classes=len(class_names))
    
    # Print a summary of our custom model using torchinfo (uncomment for actual output)
    summary(model=model,
            input_size=(128, 3,  IMG_SIZE, IMG_SIZE), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    # Setup the loss function and optimizer for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Set the seeds
    engine.set_seeds()
    print(devices)
    # Train the model and save the training results to a dictionary
    results = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=args.num_epochs,
                        work_dir=args.work_dir,
                        architecture=args.architecture,
                        device=devices)

    model_best_path = "models/" + args.architecture + "_best.pth"
    state_dict = torch.load(model_best_path,map_location=torch.device(devices))
    model_best.load_state_dict(state_dict)
    print("="*20)
    print(f"{args.architecture} Profile:")
    engine_quant.profile(model_best, test_dataloader, loss_fn, 'cpu')
