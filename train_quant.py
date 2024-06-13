from utils import data_setup, engine_quant, save
from architecture import Mobilenet_quant
from argparse import ArgumentParser
import torch
from torchinfo import summary
from torchvision import datasets, transforms


if __name__ == "__main__":
    parser = ArgumentParser(description='Train classification')
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train for.")
    parser.add_argument("--train_dir", type=str, help="Directory containing training data.", default="./data/train")
    parser.add_argument("--test_dir", type=str, help="Directory containing test data.", default="./data/test")
    parser.add_argument("--model_name", type=str, default="small", help="Model size (large or small).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use during training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate to use during training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model_dir", type=Path, default="models", help="Directory where models are saved.")
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

    model = Mobilenet_quant.MobileNetV3(args.model_name, classes=len(class_names)).to(device)

        # Print a summary of our custom model using torchinfo (uncomment for actual output)
    summary(model=model,
            input_size=(128, 3,  IMG_SIZE, IMG_SIZE), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    engine_quant.set_seeds()
    results = train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=args.num_epochs, device=device)
    
    # Save the trained model
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = args.model_dir / f"mobilenetv3_{args.model_name}_non_quantized.pth"
    torch.save(model.state_dict(), model_save_path)
    
    # Fuse and quantize model
    model = Mobilenet_quant.fuse_model(model)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Fine-tune the quantization aware model
    results = train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=args.num_epochs, device=device)
    
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    quantized_model_save_path = args.model_dir / f"mobilenetv3_{args.model_name}_quantized.pth"
    torch.save(model.state_dict(), quantized_model_save_path)
