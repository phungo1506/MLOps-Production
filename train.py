from utils import data_setup, engine, save
from architecture import MobileNetV3, ResNet
from torchinfo import summary
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train classification')
    parser.add_argument('--work-dir', default='models', help='the dir to save logs and models')
    parser.add_argument("--train-folder", default='data/train', type=str)
    parser.add_argument("--valid-folder", default='data/test', type=str)
    parser.add_argument('--architecture', action='store_true', default='MobileNetv3', help='MobileNetv3, ResNet')
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--img-size", default=112, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    args = parser.parse_args()

    print(f'Training {} model with hyper-params:'.format(args.architecture))
    train_dir = args.train_folder
    test_dir = args.valid_folder

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
                                                                                batch_size=args.batch_size)

    if args.architecture == "MobileNetv3":
        model = MobileNetV3.MobileNetV3('small')
    else:
        model = ResNet.ResNet(50)
    
    # Print a summary of our custom ViT model using torchinfo (uncomment for actual output)
    summary(model=mobinet,
            input_size=(128, 3,  112, 112), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]

    # Setup the loss function and optimizer for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mobinet.parameters(), lr=args.lr, momentum=0.9)

    # Set the seeds
    set_seeds()

    # Train the model and save the training results to a dictionary
    results_mobilenet = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=args.epochs,
                        device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir=args.work_dir,
                    model_name=args.architecture)