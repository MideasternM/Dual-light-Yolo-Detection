# -*- coding: utf-8 -*-

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import os
import tempfile
import torch
import shutil
from typing import Optional, Tuple
import sys
import base64

# --- Single-Light Model Configuration ---
try:
    from ultralytics import YOLO
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
    # Prefer a local single-light model if available
    SINGLE_MODEL_LOCAL_PATH = os.path.join(MODEL_DIR, 'visdrone_nano.pt') # Using visdrone as default single
    SINGLE_MODEL_FALLBACK = 'yolov8n.pt' # Fallback if visdrone is not found
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
            
    if os.path.exists(SINGLE_MODEL_LOCAL_PATH):
        SINGLE_MODEL_PATH = SINGLE_MODEL_LOCAL_PATH
        print(f"Loading local single-light model: {SINGLE_MODEL_PATH}")
    else:
        print(f"Local model {SINGLE_MODEL_LOCAL_PATH} not found, using fallback: {SINGLE_MODEL_FALLBACK}")
        SINGLE_MODEL_PATH = SINGLE_MODEL_FALLBACK

    single_light_model = YOLO(SINGLE_MODEL_PATH)
    print(f"Single-light model ({SINGLE_MODEL_PATH}) loaded successfully.")
except ImportError as e:
    print(f"Failed to import YOLO model from ultralytics: {e}")
    print("Please ensure ultralytics package is installed.")
    single_light_model = None
except Exception as e:
    print(f"Error loading single-light model: {e}")
    single_light_model = None

# --- Dual-Light Model Dependencies Placeholder ---
# Define placeholders for dependencies to allow Detector class definition globally
attempt_load = None
check_img_size = None
scale_coords = None
letterbox = None # Placeholder for letterbox
NMS = None
select_device = None
transformer_dependencies_loaded = False
dual_light_model_instance = None

# --- Define get_color function globally ---
def get_color(idx):
    idx = idx * 3 + 5
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

# --- Define Detector Class Globally ---
# Moved class definition outside the try-except block
class Detector:
    def __init__(self, device, model_path, model_weights, class_names, imgsz=640, merge_nms=False):
        # These dependencies are checked *before* instantiation
        global select_device, check_img_size, attempt_load
        if not all([select_device, check_img_size, attempt_load]):
             raise ImportError("Required dual-light model functions (select_device, check_img_size, attempt_load) not loaded.")

        self.device = select_device(device)
        print(f"Loading dual-light model weights: {model_weights} onto device: {self.device}")
        self.model = attempt_load(model_weights, map_location=self.device)
        self.names = class_names # Use provided class names
        print(f"Dual-light model class names set to: {self.names}")
        self.stride = max(int(self.model.stride.max()), 32)
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.merge_nms = merge_nms
        print(f"Dual-light model '{model_weights}' loaded successfully on device {self.device}.")

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocesses a single image."""
        # Use the globally loaded letterbox function
        global letterbox
        if letterbox is None:
            raise ImportError("letterbox function not loaded.")

        img0 = image.copy()
        # Note: Ensure the letterbox function signature matches (img, new_shape, stride, auto)
        # Assuming auto=True might be needed if the imported letterbox requires it.
        # Adjust call if necessary based on the specific letterbox implementation.
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() # uint8 to fp32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray, image_ir: np.ndarray, conf=0.3, iou=0.6) -> Tuple[np.ndarray, np.ndarray, list]:
        """Performs detection on RGB and IR images."""
        global NMS, scale_coords
        if not all([NMS, scale_coords]):
             raise ImportError("Required dual-light model functions (NMS, scale_coords) not loaded.")

        img_vis_rgb = image_rgb.copy()
        img_vis_ir = image_ir.copy()

        im_rgb = self._preprocess_image(image_rgb)
        im_ir = self._preprocess_image(image_ir)

        pred = self.model(im_rgb, im_ir)[0]
        pred = NMS(pred, conf_thres=conf, iou_thres=iou, classes=None, agnostic=self.merge_nms)

        processed_rgb = img_vis_rgb
        processed_ir = img_vis_ir
        detections = []

        for i, det in enumerate(pred): # detections per image
            if len(det):
                det[:, :4] = scale_coords(im_rgb.shape[2:], det[:, :4], image_rgb.shape).round()

                for *xyxy, conf_score, cls in reversed(det):
                    xmin, ymin, xmax, ymax = map(int, xyxy)
                    class_id = int(cls)
                    if class_id < 0 or class_id >= len(self.names):
                        print(f"Warning: Detected class ID {class_id} is out of bounds for known names {self.names}. Skipping.")
                        continue
                    label = f"{self.names[class_id]} {conf_score:.2f}"
                    color = get_color(class_id)

                    # Draw on RGB/IR
                    cv2.rectangle(processed_rgb, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(processed_rgb, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.rectangle(processed_ir, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(processed_ir, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    detections.append({
                        "box": [xmin, ymin, xmax, ymax],
                        "confidence": float(conf_score),
                        "class": self.names[class_id],
                        "class_id": class_id
                    })

        return processed_rgb, processed_ir, detections

# --- Dual-Light Model Configuration and Loading ---
DUAL_MODEL_DIR_NAME = 'multispectral-yolov11-transformer'
DUAL_MODEL_WEIGHTS_NAME = 'origin_x3.pt'
# --- IMPORTANT: Define the actual class names for your dual-light model here ---
# Example: DUAL_MODEL_CLASS_NAMES = ['person', 'car', 'bicycle']
DUAL_MODEL_CLASS_NAMES = [ 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ] # ADJUST THESE!
# --- /IMPORTANT ---

current_dir = os.path.dirname(__file__)
transformer_dir = os.path.join(current_dir, DUAL_MODEL_DIR_NAME)

if transformer_dir not in sys.path:
    sys.path.append(transformer_dir)
    print(f"Added {transformer_dir} to sys.path")

try:
    # Verify directory structure assumptions (optional but good practice)
    required_paths = [
        os.path.join(transformer_dir, 'models/experimental.py'),
        os.path.join(transformer_dir, 'utils/general.py'), # Still check existence
        os.path.join(transformer_dir, 'utils/torch_utils.py')
    ]
    if not os.path.isdir(transformer_dir) or not all(os.path.exists(p) for p in required_paths):
        raise FileNotFoundError(f"Required files/directories not found in {transformer_dir}")

    print(f"Attempting to import dependencies for dual-light model...")
    # Import necessary functions and assign to global placeholders
    from models.experimental import attempt_load as imported_attempt_load
    # *** Import check_img_size and scale_coords from utils.general ***
    from utils.general import check_img_size as imported_check_img_size, scale_coords as imported_scale_coords # Keep these here
    # *** Try importing NMS from ultralytics.utils.ops ***
    from ultralytics.utils.ops import non_max_suppression as imported_NMS
    # *** Import letterbox from utils.datasets ***
    from utils.datasets import letterbox as imported_letterbox # Corrected import source
    from utils.torch_utils import select_device as imported_select_device

    # Assign to global placeholders upon successful import
    attempt_load = imported_attempt_load
    check_img_size = imported_check_img_size
    scale_coords = imported_scale_coords
    letterbox = imported_letterbox # Use letterbox from utils.datasets
    NMS = imported_NMS
    select_device = imported_select_device

    transformer_dependencies_loaded = True
    print(f"Dual-light dependencies imported successfully (using letterbox from utils.datasets).") # Updated log message

    # --- Initialize Dual-Light Detector Instance ---
    DUAL_MODEL_FULL_PATH = os.path.join(transformer_dir, DUAL_MODEL_WEIGHTS_NAME)
    if os.path.exists(DUAL_MODEL_FULL_PATH):
        try:
            device_choice = 'cpu' # or 'cuda:0'
            # Pass necessary components to the constructor
            dual_light_model_instance = Detector(
                device=device_choice,
                model_path=transformer_dir, # Pass base path if needed by attempt_load internally
                model_weights=DUAL_MODEL_FULL_PATH,
                class_names=DUAL_MODEL_CLASS_NAMES, # Pass the defined names
                imgsz=640
            )
        except Exception as e:
            print(f"ERROR: Failed to initialize dual-light Detector: {e}")
            dual_light_model_instance = None
            transformer_dependencies_loaded = False # Mark as failed
    else:
        print(f"ERROR: Dual-light model weights not found at: {DUAL_MODEL_FULL_PATH}")
        dual_light_model_instance = None
        transformer_dependencies_loaded = False

except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Dual-light model setup skipped.")
except ImportError as e:
    print(f"ERROR: Failed to import dependencies for dual-light model: {e}")
    print(f"Please ensure '{DUAL_MODEL_DIR_NAME}' is correctly structured and all sub-dependencies are installed.")
    transformer_dependencies_loaded = False
except Exception as e:
    print(f"ERROR: An unexpected error occurred during dual-light model setup: {e}")
    transformer_dependencies_loaded = False

app = FastAPI(title="YOLO Object Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

temp_dir = tempfile.mkdtemp()

@app.on_event("shutdown")
def cleanup():
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Cleaned up temp directory.")

@app.get("/")
def read_root():
    return {"message": "YOLO Object Detection API. Use /detect endpoint."}


# --- Updated Detect Endpoint ---
@app.post("/detect")
async def detect_objects(
    mode: str = Form(...),
    file_type: str = Form(...),
    file_rgb: UploadFile = File(...),
    file_ir: Optional[UploadFile] = File(None)
):
    print(f"Received request: mode='{mode}', file_type='{file_type}', rgb_filename='{file_rgb.filename}', ir_filename='{file_ir.filename if file_ir else 'N/A'}'")

    if mode == "single":
        if single_light_model is None:
            raise HTTPException(status_code=503, detail="Single-light model not loaded.")
        if file_type == "image":
            print("Processing single-light image...")
            return await process_single_image(file_rgb, single_light_model)
        elif file_type == "video":
            print("Processing single-light video...")
            return await process_single_video(file_rgb, single_light_model)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for single mode.")

    elif mode == "dual":
        if dual_light_model_instance is None or not transformer_dependencies_loaded:
            error_detail = "Dual-light model is not available."
            if dual_light_model_instance is None and transformer_dependencies_loaded:
                 error_detail = "Dual-light model weights found, but Detector initialization failed."
            elif not transformer_dependencies_loaded:
                 error_detail = "Dual-light model dependencies failed to load."
            raise HTTPException(status_code=503, detail=error_detail)

        if file_type == "video":
            raise HTTPException(status_code=400, detail="Dual-light video processing is not yet supported.")
        elif file_type == "image":
            if file_ir is None:
                raise HTTPException(status_code=400, detail="Dual mode requires both 'file_rgb' and 'file_ir' image uploads.")
            print("Processing dual-light image...")
            return await process_dual_image(file_rgb, file_ir, dual_light_model_instance)
        else:
             raise HTTPException(status_code=400, detail="Unsupported file type for dual mode.")

    else:
        raise HTTPException(status_code=400, detail="Invalid mode specified. Use 'single' or 'dual'.")


# --- Refactored Processing Functions ---

async def process_single_image(file: UploadFile, model):
    """Processes a single image using the provided YOLO model."""
    temp_image_path = os.path.join(temp_dir, f"temp_image_{file.filename}")
    try:
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
        results = model(temp_image_path) # Use the passed model
    
        if not results or len(results) == 0:
                raise HTTPException(status_code=500, detail="No detection results returned by single-light model.")
        
        result = results[0]
        result_img = result.plot() # Use plot() method for visualization
        
        if result_img is None or result_img.size == 0:
            raise HTTPException(status_code=500, detail="Empty detection results after plotting.")
            result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        # result_img is usually RGB from ultralytics, convert to BGR for cv2.imencode
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

        _, encoded_img = cv2.imencode('.jpg', result_img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        print(f"Error processing single image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing single image: {str(e)}")
    finally:
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e:
                    print(f"Failed to clean up temp image file {temp_image_path}: {e}")

async def process_single_video(file: UploadFile, model):
    """Processes a single video using the provided YOLO model."""
    temp_video_path = os.path.join(temp_dir, f"temp_video_{file.filename}")
    temp_output_path = os.path.join(temp_dir, f"output_{os.path.splitext(file.filename)[0]}.mp4") # Ensure .mp4 extension
    
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Unable to open video file.")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1 # Avoid division by zero for short/empty videos

        # Use H.264 codec ('avc1')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        if not out.isOpened():
             raise HTTPException(status_code=500, detail=f"Failed to initialize video writer with codec 'avc1'. Check if codec is supported.")
    
        processed_frames = 0
        print(f"Starting video processing: {total_frames} frames, {fps} FPS, {width}x{height}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                    # Perform detection frame by frame
                results = model(frame) # Use the passed model
                rendered_frame = results[0].plot()
                    # Convert RGB plot back to BGR for VideoWriter
                out.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))

                processed_frames += 1
                if processed_frames % max(1, int(total_frames / 20)) == 0: # Print progress roughly 20 times
                    progress = processed_frames / total_frames
                    print(f"Video processing progress: {progress:.1%} ({processed_frames}/{total_frames})")
                
            except Exception as e:
                    print(f"Error processing frame {processed_frames}: {e}")
                    # Continue processing other frames if one fails
                    continue
        
            print(f"Finished processing video. Total frames processed: {processed_frames}")
        cap.release()
        out.release()
        
        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
                raise HTTPException(status_code=500, detail="Video processing failed - output file is empty or not created.")
        
        def iterfile():
            try:
                with open(temp_output_path, "rb") as f:
                        while chunk := f.read(1024 * 1024): # 1MB chunks
                            yield chunk
            finally:
                    # Clean up output file after streaming is done
                if os.path.exists(temp_output_path):
                        try:
                            os.remove(temp_output_path)
                            print(f"Cleaned up temp output file: {temp_output_path}")
                        except Exception as e:
                            print(f"Error cleaning up temp output file {temp_output_path}: {e}")

        
        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                    "Content-Disposition": f"inline; filename=\"output_{file.filename}\"", # Use a generic output name
            "Content-Type": "video/mp4"
        }
        )

    except Exception as e:
        print(f"Error processing single video: {e}")
        # Ensure resources are released even if an error occurs mid-processing
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'out' in locals() and out.isOpened(): out.release()
        raise HTTPException(status_code=500, detail=f"Error processing single video: {str(e)}")
    finally:
        # Clean up input file
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception as e:
                print(f"Failed to clean up temp video file {temp_video_path}: {e}")
        # Output file is cleaned up in iterfile's finally block or if error occurred before streaming


async def process_dual_image(file_rgb: UploadFile, file_ir: UploadFile, detector: Detector):
    """Processes RGB and IR images using the dual-light Detector."""
    temp_rgb_path = os.path.join(temp_dir, f"temp_rgb_{file_rgb.filename}")
    temp_ir_path = os.path.join(temp_dir, f"temp_ir_{file_ir.filename}")
    try:
        with open(temp_rgb_path, "wb") as buffer:
            shutil.copyfileobj(file_rgb.file, buffer)
        with open(temp_ir_path, "wb") as buffer:
            shutil.copyfileobj(file_ir.file, buffer)

        image_rgb = cv2.imread(temp_rgb_path)
        image_ir = cv2.imread(temp_ir_path)

        if image_rgb is None:
            raise HTTPException(status_code=400, detail=f"Could not read RGB image file: {file_rgb.filename}")
        if image_ir is None:
            raise HTTPException(status_code=400, detail=f"Could not read IR image file: {file_ir.filename}")

        processed_rgb, processed_ir, detections = detector(image_rgb, image_ir)
        print(f"Dual-light detection complete. Found {len(detections)} objects.")

        # Encode both images to JPEG bytes
        ret_rgb, encoded_rgb = cv2.imencode('.jpg', processed_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ret_ir, encoded_ir = cv2.imencode('.jpg', processed_ir, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Check if encoding was successful
        if not ret_rgb or not ret_ir:
             raise HTTPException(status_code=500, detail="Failed to encode processed images.")

        # Base64 encode the bytes
        base64_rgb = base64.b64encode(encoded_rgb.tobytes()).decode('utf-8')
        base64_ir = base64.b64encode(encoded_ir.tobytes()).decode('utf-8')

        # Return JSON response with Base64 strings and detections
        return JSONResponse(content={
            "rgb_image": base64_rgb,
            "ir_image": base64_ir,
            "detections": detections
        })

    except Exception as e:
        print(f"Error processing dual image: {e}")
        # Consider adding traceback here for debugging if needed
        # import traceback
        # traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing dual image: {str(e)}")
    finally:
        # Clean up temporary files
        for p in [temp_rgb_path, temp_ir_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"Failed to clean up temp file {p}: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    # Ensure reload=True is suitable for production? Maybe False.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)