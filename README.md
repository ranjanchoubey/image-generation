# Image Face Detection and Cropping Tool

## What Does This Tool Do?

This tool takes a folder of photos, finds faces in each photo, and creates new cropped photos that focus on the detected faces. Think of it like automatic portrait mode - it finds a person's face and crops the image around it.

![Example: Original image → Cropped face image](https://i.imgur.com/example.png)

## Before You Begin

This project requires some software to be installed on your computer. Don't worry - we'll walk through each step!

### For Complete Beginners: Opening a Terminal

To run the commands in this guide, you'll need to use a terminal (also called command line or console):

- **Windows**: Press `Win+R`, type `cmd` and press Enter. Or search for "Command Prompt" in the Start menu.
- **Mac**: Press `Cmd+Space`, type "Terminal" and press Enter.
- **Linux**: Press `Ctrl+Alt+T` or find "Terminal" in your applications menu.

When you see a command in a gray box like this:
```
some command here
```
You need to type it into the terminal and press Enter to run it.

### Step 1: Installing Rust Programming Language

Rust is the programming language used to build this tool. Here's how to install it:

1. Open a terminal window
2. Copy and paste this command and press Enter:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
3. When prompted, press `1` and Enter to choose the default installation
4. After installation completes, type or copy this command and press Enter:
   ```bash
   source $HOME/.cargo/env
   ```
5. To make sure Rust installed correctly, run these commands:
   ```bash
   rustc --version
   cargo --version
   ```
   You should see version information, not error messages

If you get "command not found" errors, close your terminal, open a new one, and try the verification commands again.

### Step 2: Installing Required Dependencies

This tool relies on other software libraries to work properly. Run these commands to install them:

#### For Ubuntu/Linux Mint/Debian:
```bash
# This updates your system's package list
sudo apt update

# This installs the OpenCV library for image processing
sudo apt install libopencv-dev clang libclang-dev

# This installs a library needed for the neural network model
sudo apt install libgomp1
```
When asked for your password, type it (it won't show as you type) and press Enter.

#### For Windows:
Windows installation is more complex. We recommend using WSL (Windows Subsystem for Linux) and following the Ubuntu instructions above.

#### For Mac:
```bash
# Install Homebrew package manager if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install opencv libomp
```

### Step 3: Getting the Project Files

1. If you haven't already, install Git:
   ```bash
   # Ubuntu/Debian
   sudo apt install git
   
   # macOS
   brew install git
   ```

2. Download the project:
   ```bash
   git clone https://github.com/ranjankumarchoubey/image-generation.git
   cd image-generation
   ```
   This creates a folder called "image-generation" with all the project files.

### Step 4: Building the Software

Now let's build the program. This may take a few minutes the first time.

```bash
cargo build --release
```

You should see a lot of text output as it builds. It's done when you see your command prompt again.

## Getting a Face Detection Model

The tool needs a pre-trained AI model to detect faces in images. Here's how to get one:

### Option 1: Download a Pre-Converted Model (Easiest)

1. Download this pre-converted model file:
   ```bash
   curl -LO https://example.com/model.pt
   ```
   (Note: Replace with actual URL when you have one)

2. Make sure the file is in your "image-generation" folder

### Option 2: Converting Your Own Model (Advanced)

If you're comfortable with Python, you can convert a model yourself:

1. Install Python and PyTorch:
   ```bash
   # Ubuntu/Debian
   sudo apt install python3 python3-pip
   pip3 install torch torchvision
   ```

2. Create a Python script called `convert_model.py`:
   ```python
   import torch
   model = torch.hub.load('deepcam-cn/yolov5-face', 'yolov5s_face', pretrained=True)
   script_model = torch.jit.script(model)
   script_model.save("model.pt")
   ```

3. Run the script:
   ```bash
   python3 convert_model.py
   ```
   This will download and convert the model, which might take several minutes. You'll see "model.pt" in your folder when it's done.

## Using the Face Detection Tool

### Preparing Your Images

1. Create two folders to store your images:
   ```bash
   mkdir -p input_images cropped_faces
   ```

2. Copy the photos you want to process into the "input_images" folder:
   ```bash
   # Example: if your photos are in Pictures/family
   cp ~/Pictures/family/*.jpg input_images/
   ```
   Only JPG, JPEG, and PNG files will be processed.

### Running the Tool

Now you're ready to run the tool:

```bash
./target/release/image-generation --input-dir ./input_images --output-dir ./cropped_faces
```

This tells the program to:
- Look for photos in the "input_images" folder
- Save the cropped face photos to the "cropped_faces" folder
- Use default settings for everything else

You'll see messages for each processed photo. The program will:
- Only process images with exactly one face
- Skip images where it can't find a face or finds multiple faces

### Additional Options

For better results, you can adjust the sensitivity of face detection:

```bash
./target/release/image-generation --input-dir ./input_images --output-dir ./cropped_faces --conf-thresh 0.4
```

The `--conf-thresh` value ranges from 0 to 1:
- Lower values (like 0.3) will detect more faces but might include false detections
- Higher values (like 0.7) are stricter, detecting only very clear faces

If you put the model file somewhere else, tell the program where to find it:

```bash
./target/release/image-generation --input-dir ./input_images --output-dir ./cropped_faces --model-path /path/to/your/model.pt
```

## Troubleshooting

### "Command not found" errors
- Make sure you're in the right folder. Type `pwd` to check.
- For `cargo` errors, try closing and reopening your terminal after installing Rust.

### "Cannot find model.pt" errors
- Make sure the model file is in the same folder as the program.
- Use the `--model-path` option to specify the correct location.

### No faces detected in your images
- Try reducing the confidence threshold: `--conf-thresh 0.3`
- Make sure faces in the images are reasonably clear and not too small.

### Program crashes or other errors
- Make sure all dependencies are properly installed.
- For specific error messages, please search online or ask for help in our project's discussion forum.

## Glossary of Terms

- **Terminal/Command Line**: Text interface to interact with your computer
- **Rust**: Programming language used to build this tool
- **Cargo**: Rust's package manager and build tool
- **OpenCV**: Computer vision library for image processing
- **Model**: Pre-trained AI that can recognize patterns (in our case, faces)
- **TorchScript**: Format for the AI model
- **Confidence threshold**: How certain the AI must be before confirming a face detection

## Notes

- The application only processes images with a single detected face meeting the confidence threshold
- Supported image formats: JPG, JPEG, PNG
- The face cropping uses a scaling factor of 2.0 around the detected face bounding box, meaning it includes some surrounding area

