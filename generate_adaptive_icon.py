import os
from PIL import Image
from rembg import remove

def process_icon():
    input_path = "mobile/assets/images/hydra-logo.png"
    output_path = "mobile/assets/images/adaptive-icon.png"
    
    # Read the image
    with open(input_path, 'rb') as i:
        input_image = i.read()
    
    from rembg import new_session
    
    # Remove the background using a smaller model (u2netp) for speed
    print("Removing background...")
    session = new_session("u2netp")
    subject = remove(input_image, session=session)
    
    # Save a temporary transparent image
    temp_path = "mobile/assets/images/temp_transparent.png"
    with open(temp_path, 'wb') as o:
        o.write(subject)
        
    # Open the transparent image
    with Image.open(temp_path) as img:
        # Convert to RGBA if not already
        img = img.convert("RGBA")
        
        # Get bounding box of the non-transparent area
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            
        print(f"Original cropped size: {img.size}")
        
        # Calculate adaptive icon sizes (1024x1024 canvas, logo 624x624 max)
        canvas_size = 1024
        max_logo_size = 624
        
        # Calculate scaling to fit within max_logo_size while keeping aspect ratio
        ratio = min(max_logo_size / img.width, max_logo_size / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a blank transparent canvas
        canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
        
        # Paste the resized logo in the center
        paste_x = (canvas_size - new_width) // 2
        paste_y = (canvas_size - new_height) // 2
        
        # Need to use the image itself as the mask to preserve transparency
        canvas.paste(img_resized, (paste_x, paste_y), img_resized)
        
        # Save the final adaptive icon
        canvas.save(output_path, format="PNG")
        print(f"Successfully saved adaptive icon to {output_path}")

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
if __name__ == "__main__":
    process_icon()
