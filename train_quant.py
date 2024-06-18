from utils import data_setup, engine_quant, save
from architecture import Mobilenet, Resnet, Efficientnet
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

    if args.architecture == "MobileNetv3":
        model = Mobilenet.MobileNetV3(config_name='small', num_classes=len(class_names))
    if args.architecture == "EfficientNet":
        model = Efficientnet.EfficientNet(version='b1', num_classes=len(class_names))
    if args.architecture == "Resnet":
        model = Resnet.ResNet50(num_classes=len(class_names))
    
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
    engine_quant.set_seeds()
    # Train the model and save the training results to a dictionary
    results = engine_quant.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=args.num_epochs,
                        work_dir=args.work_dir,
                        architecture=args.architecture,
                        device=devices)

    print("MobileNetv3 Profile:")
    engine_quant.profile(model, test_dataloader, loss_fn, 'cpu')
    example_inputs, classes = next(iter(test_dataloader))  
    model.eval()  # Set the model to evaluation mode
    model.to('cpu')

    # Try Dynamic Quantization
    dynamic_qconfig = torch.quantization.default_dynamic_qconfig
    qconfig_dict = {
        # Global Config
        "": dynamic_qconfig
    }
    # Assuming resnet expects an input tensor of shape [1, 3, 224, 224] with float32 dtype
    model_prepared = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)  # Pass example_inputs to prepare_fx
    dynamic_resnet = convert_fx(model_prepared)
    print("MobileNetv3 Dynamic-Quant Profile:")
    engine_quant.profile(dynamic_resnet, test_dataloader, loss_fn, 'cpu')

    # Try Static Quantization
    static_qconfig = torch.quantization.get_default_qconfig('fbgemm')
    qconfig_dict = {
        # Global Config
        "": static_qconfig,
    }
    mp = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
    mc = convert_fx(mp)
    engine_quant.profile(mc, test_dataloader, loss_fn, 'cpu')