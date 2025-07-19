from pathlib import Path
import sys
import pandas as pd
import cv2
import shutil
from tqdm import tqdm
from collections import defaultdict
import gc
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import logging 

# Config log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_local_time():
    return datetime.now(ZoneInfo("America/New_York"))


class SimpleTimer:
    def __init__(self):
        self.start_time = time.time()
        
    def elapsed(self):
        seconds = time.time() - self.start_time
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.1f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def checkpoint(self, message):
        elapsed_time = self.elapsed()
        print(f"{message} (Elapsed: {elapsed_time})")
        logging.info(f"{message} (Elapsed: {elapsed_time})")


# Initialize timer
timer = SimpleTimer()

# Actual Configuration
RESIZE_MODE = "small"
BATCH_SIZE = 50  # Process images in batches to manage memory
JPEG_QUALITY = 95  # (1-100)

# AI helped me to discover the below edge case. 
# Configuration for input file extensions and grayscale handling
INPUT_IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.dcm"] # List of extensions to process
CONVERT_GRAYSCALE_TO_RGB = True 

# Validate JPEG_QUALITY
if not (1 <= JPEG_QUALITY <= 100):
    raise ValueError("JPEG_QUALITY must be between 1 and 100.")

# These images are 512x512px, so this should be a 25% and 50% decrease in overall storage
resize_settings = {
    "small": (128, 128),
    "medium": (256, 256)
}

resize_dim = resize_settings.get(RESIZE_MODE, (256, 256))
# Validate RESIZE_MODE
if RESIZE_MODE not in resize_settings:
    logging.warning(f"Warning: RESIZE_MODE '{RESIZE_MODE}' not recognized. Defaulting to {resize_dim}.")


dataset_root = Path("/kaggle/input/hnscc-zip/HNSCC_data")
ct_images_path = dataset_root / "ct_images"
clinical_csv_path = dataset_root / "clinical.csv"

# Output with resize information to allow for multiple runs of the script, by size.
resize_suffix = f"{resize_dim[0]}x{resize_dim[1]}"
output_root = Path(f"/kaggle/working/Updated_Data_{resize_suffix}_{RESIZE_MODE}")

def print_table(data, headers, title="", csv_path=None):
    """Print formatted table and optionally save to CSV"""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Print table to console
    print(tabulate(data, headers=headers, tablefmt="grid", floatfmt=".1f"))
    
    
    if csv_path:
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(csv_path, index=False)
        print(f"Table saved to: {csv_path}")
        logging.info(f"Table '{title}' saved to: {csv_path}")

# Raise error instead of printing the error message from earlier versions.
if not ct_images_path.exists():
    logging.error(f"The path {ct_images_path} does not exist.")
    raise FileNotFoundError(f"The path {ct_images_path} does not exist.")
if not clinical_csv_path.exists():
    logging.error(f"The path {clinical_csv_path} does not exist.")
    raise FileNotFoundError(f"The path {clinical_csv_path} does not exist.")

print(f"Execution started at: {get_local_time().strftime('%Y-%m-%d %H:%M:%S')}")

logging.infoprint(f"Execution started at: {get_local_time().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {output_root}")
print(f"Processing configuration: {RESIZE_MODE} mode, {resize_dim}, Quality: {JPEG_QUALITY}")
print(f"Input image extensions: {', '.join(INPUT_IMAGE_EXTENSIONS)}")
print(f"Convert grayscale to RGB: {CONVERT_GRAYSCALE_TO_RGB}")


# I updated the overall folder structure here to include the new logs.
csv_to_folder_dir = output_root / "csv_to_folder_check"
folder_to_csv_dir = output_root / "folder_to_csv_check"
missing_data_dir = output_root / "missing_data"
processing_logs_dir = output_root / "processing_logs"
tables_dir = output_root / "summary_tables"
visualizations_dir = output_root / "visualizations"
error_logs_dir = output_root / "error_logs" 

for path in [csv_to_folder_dir, folder_to_csv_dir, missing_data_dir, processing_logs_dir, tables_dir, visualizations_dir, error_logs_dir]:
    path.mkdir(parents=True, exist_ok=True)

# AI helped discover the need for an error log file, and its construction.
error_log_file = error_logs_dir / "image_processing_errors.log"
file_handler = logging.FileHandler(error_log_file)
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler) # Add file handler to root logger

print("Loading my dataset...")
logging.info("Loading my dataset...")
folder_ids = {p.name.replace("_ct_images", "") for p in ct_images_path.iterdir() if p.is_dir()}
df = pd.read_csv(clinical_csv_path)
csv_ids = set(df["TCIA Radiomics dummy ID of To_Submit_Final"].astype(str).str.strip())


n_folders = len(folder_ids)
n_csv_ids = len(csv_ids)

timer.checkpoint("Data Loading Complete")

# CSV to Folder check
print(f"Performing data matching analysis...")
logging.info("Performing data matching analysis...")
csv_to_folder_results = []
missing_csv_to_folder = []

for idx, csv_id in enumerate(csv_ids, start=1):
    match_found = csv_id in folder_ids
    status = "Confirmed" if match_found else "Missing"
    
    csv_to_folder_results.append([idx, csv_id, match_found, status])
    
    if not match_found:
        missing_csv_to_folder.append(csv_id)
        logging.warning(f"CSV ID '{csv_id}' has no corresponding image folder.")


# Folder to CSV check
folder_to_csv_results = []
missing_folder_to_csv = []

for idx, folder_id in enumerate(folder_ids, start=1):
    match_found = folder_id in csv_ids
    status = "Confirmed" if match_found else "Missing"
    
    folder_to_csv_results.append([idx, folder_id, match_found, status])
    
    if not match_found:
        missing_folder_to_csv.append(folder_id)
        logging.warning(f"Image folder '{folder_id}' has no corresponding CSV entry.")


# Print matching analysis tables
print_table(
    csv_to_folder_results[:10],  # This shows the first 10.
    ["Patient Number", "CSV ID", "Match Found", "Status"],
    "CSV to Folder Matching Analysis (First 10)",
    tables_dir / "csv_to_folder_analysis.csv"
)

print_table(
    folder_to_csv_results[:10], 
    ["Folder Number", "Folder ID", "Match Found", "Status"],
    "Folder to CSV Matching Analysis (First 10)",
    tables_dir / "folder_to_csv_analysis.csv"
)

# Simple Summary Table
matched_count = len(folder_ids.intersection(csv_ids))
match_rate = (matched_count / max(n_csv_ids, 1)) * 100

data_quality_summary = [
    ["Total CSV Records", n_csv_ids],
    ["Total Image Folders", n_folders],
    ["Matched Pairs", matched_count],
    ["CSV Missing Folders", len(missing_csv_to_folder)],
    ["Folders Missing CSV", len(missing_folder_to_csv)],
    ["Match Rate Percent", f"{match_rate:.1f}"]
]

print_table(
    data_quality_summary,
    ["Metric", "Count"],
    "DATA QUALITY SUMMARY",
    tables_dir / "data_quality_summary.csv"
)

# Results
pd.DataFrame(csv_to_folder_results, columns=["Patient Number", "CSV ID", "Match Found", "Status"]).to_csv(csv_to_folder_dir / "matched_and_unmatched.csv", index=False)
pd.DataFrame(folder_to_csv_results, columns=["Folder Number", "Folder ID", "Match Found", "Status"]).to_csv(folder_to_csv_dir / "matched_and_unmatched.csv", index=False)
pd.DataFrame({"Missing CSV IDs (No Folder)": missing_csv_to_folder}).to_csv(missing_data_dir / "csv_without_folder_match.csv", index=False)
pd.DataFrame({"Missing Folder IDs (No CSV)": missing_folder_to_csv}).to_csv(missing_data_dir / "folders_without_csv_match.csv", index=False)


matched_output_dir = output_root / f"matched_folders_{resize_suffix}"
matched_output_dir.mkdir(parents=True, exist_ok=True)

# Create list of matched patient IDs
matched_ids = folder_ids.intersection(csv_ids)

timer.checkpoint("Matching Analysis Complete")

# Initialize tracking dict
processing_stats = defaultdict(lambda: {
    'total_images': 0,
    'successfully_processed': 0,
    'skipped_images': [],
    'error_messages': [],
    'processing_time': 0,
    'batch_count': 0
})

overall_stats = {
    'total_images_found': 0,
    'total_successfully_processed': 0,
    'total_skipped': 0,
    'patients_processed': 0,
    'total_processing_time': 0
}

def process_image_batch(image_paths, dest_folder, patient_id):
    """Process a batch of images for memory efficiency"""
    batch_start = time.time()
    batch_stats = {'processed': 0, 'skipped': []}
    
    for image_file in image_paths:
        try:
            # Try block to check for grayscale first.
            img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                # AI helped me to find this edge case.
                # If grayscale read fails, try reading as color.
                img = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
                if img is None:
                    error_msg = f"Could not read image: {image_file.name}. File might be corrupted or unsupported."
                    processing_stats[patient_id]['skipped_images'].append(image_file.name)
                    processing_stats[patient_id]['error_messages'].append(error_msg)
                    batch_stats['skipped'].append(image_file.name)
                    logging.error(f"Patient {patient_id}: {error_msg}")
                    continue

            # Image resize
            resized = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)

            # Convert to RGB/BGR if specified and necessary.
            if CONVERT_GRAYSCALE_TO_RGB:
                if len(resized.shape) == 2: # It's grayscale.
                    img_output = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR) # Conversion here
                else: # Handles if image is already 3-channel.
                    img_output = resized
            else: # Do not force grayscale to RGB, save as is. 
                img_output = resized

            # Save as configurable quality JPG/JPEG.
            output_filename = image_file.stem + ".jpg"
            output_path = dest_folder / output_filename
            cv2.imwrite(str(output_path), img_output, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            
            batch_stats['processed'] += 1
            
        except Exception as e:
            error_msg = f"Error processing {image_file.name}: {str(e)}"
            processing_stats[patient_id]['skipped_images'].append(image_file.name)
            processing_stats[patient_id]['error_messages'].append(error_msg)
            batch_stats['skipped'].append(image_file.name)
            logging.error(f"Patient {patient_id}: {error_msg}")
    
    # Waste Management (garbage collection) 
    gc.collect()
    return batch_stats

print(f"\nSTARTING IMAGE PROCESSING")
logging.info("STARTING IMAGE PROCESSING")
print(f"Configuration: {RESIZE_MODE} {resize_dim} | JPEG Quality: {JPEG_QUALITY} | Batch Size: {BATCH_SIZE}")
print("=" * 80)

processing_start_time = time.time()

# Process patients
patient_summary_data = []

for patient_id in tqdm(matched_ids, desc="Processing patients"):
    patient_start_time = time.time()
    
    src_folder = ct_images_path / f"{patient_id}_ct_images"
    dest_folder = matched_output_dir / f"{patient_id}_ct_images"
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all image files for this patient
    image_files = []
    for ext in INPUT_IMAGE_EXTENSIONS:
        image_files.extend(list(src_folder.glob(ext)))
    
    processing_stats[patient_id]['total_images'] = len(image_files)
    overall_stats['total_images_found'] += len(image_files)
    
    if not image_files:
        logging.warning(f"Patient {patient_id}: No image files found with extensions {INPUT_IMAGE_EXTENSIONS}.")
        continue
    
    # Process images in batches
    batch_count = 0
    for i in range(0, len(image_files), BATCH_SIZE):
        batch = image_files[i:i + BATCH_SIZE]
        batch_count += 1
        
        batch_results = process_image_batch(batch, dest_folder, patient_id)
        processing_stats[patient_id]['successfully_processed'] += batch_results['processed']
    
    patient_processing_time = time.time() - patient_start_time
    processing_stats[patient_id]['processing_time'] = patient_processing_time
    processing_stats[patient_id]['batch_count'] = batch_count
    
    # Update overall stats
    overall_stats['total_successfully_processed'] += processing_stats[patient_id]['successfully_processed']
    overall_stats['total_skipped'] += len(processing_stats[patient_id]['skipped_images'])
    overall_stats['patients_processed'] += 1
    
    # Collect data for summary table
    success_rate = (processing_stats[patient_id]['successfully_processed'] / max(processing_stats[patient_id]['total_images'], 1)) * 100
    
    patient_summary_data.append([
        patient_id,
        processing_stats[patient_id]['total_images'],
        processing_stats[patient_id]['successfully_processed'],
        len(processing_stats[patient_id]['skipped_images']),
        f"{success_rate:.1f}",
        f"{patient_processing_time:.2f}",
        batch_count
    ])
    logging.info(f"Patient {patient_id} processed. Total images: {processing_stats[patient_id]['total_images']}, Processed: {processing_stats[patient_id]['successfully_processed']}, Skipped: {len(processing_stats[patient_id]['skipped_images'])}")


overall_stats['total_processing_time'] = time.time() - processing_start_time
timer.checkpoint("Image Processing Complete")

# Patient Processing Summary Table
print_table(
    patient_summary_data[:15],  # Show first 15.
    ["Patient ID", "Total Images", "Processed", "Skipped", "Success Rate Percent", "Time Seconds", "Batches"],
    "PATIENT PROCESSING SUMMARY (First 15)",
    tables_dir / "patient_processing_summary.csv"
)

# Overall Processing Statistics Table
avg_time_per_patient = overall_stats['total_processing_time'] / max(overall_stats['patients_processed'], 1)
images_per_second = overall_stats['total_successfully_processed'] / max(overall_stats['total_processing_time'], 1)
overall_success_rate = (overall_stats['total_successfully_processed'] / max(overall_stats['total_images_found'], 1)) * 100

processing_summary = [
    ["Patients Processed", overall_stats['patients_processed']],
    ["Total Images Found", overall_stats['total_images_found']],
    ["Successfully Processed", overall_stats['total_successfully_processed']],
    ["Images Skipped", overall_stats['total_skipped']],
    ["Overall Success Rate Percent", f"{overall_success_rate:.1f}"],
    ["Total Processing Time Seconds", f"{overall_stats['total_processing_time']:.1f}"],
    ["Average Time per Patient Seconds", f"{avg_time_per_patient:.2f}"],
    ["Images per Second", f"{images_per_second:.1f}"]
]

print_table(
    processing_summary,
    ["Metric", "Value"],
    "OVERALL PROCESSING STATISTICS",
    tables_dir / "overall_processing_stats.csv"
)

# Create Summary CSV
image_count_summary = {
    'Configuration': f"{RESIZE_MODE}_{resize_suffix}",
    'Resize_Dimensions': f"{resize_dim[0]}x{resize_dim[1]}",
    'JPEG_Quality': JPEG_QUALITY,
    'Batch_Size': BATCH_SIZE,
    'Input_Extensions': '; '.join(INPUT_IMAGE_EXTENSIONS),
    'Convert_Grayscale_to_RGB': CONVERT_GRAYSCALE_TO_RGB,
    'Total_Patients_Matched': len(matched_ids),
    'Total_Images_Found': overall_stats['total_images_found'],
    'Total_Images_Processed': overall_stats['total_successfully_processed'],
    'Total_Images_Skipped': overall_stats['total_skipped'],
    'Success_Rate_Percent': f"{overall_success_rate:.1f}",
    'Processing_Time_Seconds': f"{overall_stats['total_processing_time']:.1f}",
    'Total_Execution_Time': timer.elapsed(),
    'Output_Directory': str(output_root),
    'Processed_Images_Directory': str(matched_output_dir),
    'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

summary_df = pd.DataFrame([image_count_summary])
summary_csv_path = output_root / "execution_summary.csv"
summary_df.to_csv(summary_csv_path, index=False)

print_table(
    [[k, v] for k, v in image_count_summary.items()],
    ["Configuration Item", "Value"],
    "EXECUTION SUMMARY",
    summary_csv_path
)

# Create Visualizations
timer.checkpoint("Creating performance visualizations")

# 1. Success Rate Distribution
plt.figure(figsize=(12, 8))
success_rates = [float(row[4]) for row in patient_summary_data]

plt.subplot(2, 2, 1)
plt.hist(success_rates, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Success Rates Across Patients')
plt.xlabel('Success Rate (%)')
plt.ylabel('Number of Patients')

# 2. Processing Time vs Images
plt.subplot(2, 2, 2)
processing_times = [float(row[5]) for row in patient_summary_data]
total_images = [int(row[1]) for row in patient_summary_data]
plt.scatter(total_images, processing_times, alpha=0.6, color='green')
plt.title('Processing Time vs Number of Images')
plt.xlabel('Total Images per Patient')
plt.ylabel('Processing Time (seconds)')

# 3. Images per Patient Distribution
plt.subplot(2, 2, 3)
plt.hist(total_images, bins=15, color='orange', alpha=0.7, edgecolor='black')
plt.title('Distribution of Images per Patient')
plt.xlabel('Number of Images')
plt.ylabel('Number of Patients')

# 4. Overall Statistics Pie Chart
plt.subplot(2, 2, 4)
labels = ['Successfully Processed', 'Skipped']
sizes = [overall_stats['total_successfully_processed'], overall_stats['total_skipped']]
colors = ['lightgreen', 'lightcoral']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Image Processing Results')

plt.tight_layout()
plt.savefig(visualizations_dir / "processing_performance_dashboard.png", dpi=300, bbox_inches='tight')
plt.show()

# Save all detailed reports
processing_report = []
for patient_id, stats in processing_stats.items():
    processing_report.append({
        'Patient_ID': patient_id,
        'Total_Images': stats['total_images'],
        'Successfully_Processed': stats['successfully_processed'],
        'Skipped_Count': len(stats['skipped_images']),
        'Processing_Time_Seconds': stats['processing_time'],
        'Batch_Count': stats['batch_count'],
        'Success_Rate_Percent': f"{(stats['successfully_processed'] / max(stats['total_images'], 1)) * 100:.1f}",
        'Images_Per_Second': f"{stats['successfully_processed'] / max(stats['processing_time'], 1):.2f}",
        'Skipped_Images': '; '.join(stats['skipped_images']) if stats['skipped_images'] else 'None',
        'Error_Messages': '; '.join(stats['error_messages']) if stats['error_messages'] else 'None'
    })

processing_df = pd.DataFrame(processing_report)
processing_df.to_csv(processing_logs_dir / "detailed_processing_report.csv", index=False)

# Filter and save matched CSV
matched_df = df[df["TCIA Radiomics dummy ID of To_Submit_Final"].astype(str).str.strip().isin(matched_ids)]
matched_df.to_csv(output_root / "matched_clinical_data.csv", index=False)

# Final Summary Table
final_summary = [
    ["Data Files", "", ""],
    ["Matched Clinical CSV", str(output_root / "matched_clinical_data.csv"), "Complete"],
    ["Resized Images Folder", str(matched_output_dir), "Complete"],
    ["Analysis Tables", "", ""],
    ["Data Quality Summary", str(tables_dir / "data_quality_summary.csv"), "Complete"],
    ["Patient Processing Summary", str(tables_dir / "patient_processing_summary.csv"), "Complete"],
    ["Execution Summary", str(summary_csv_path), "Complete"],
    ["Visualizations", "", ""],
    ["Performance Dashboard", str(visualizations_dir / "processing_performance_dashboard.png"), "Complete"],
    ["Detailed Reports", "", ""],
    ["Processing Log", str(processing_logs_dir / "detailed_processing_report.csv"), "Complete"],
    ["Detailed Error Log", str(error_log_file), "Complete"], # New: Error log entry
    ["All Output Files", str(output_root), "Complete"]
]

print_table(
    final_summary,
    ["Category", "Location", "Status"],
    "FINAL OUTPUT SUMMARY",
    tables_dir / "final_output_summary.csv"
)

# Zip w/ resize info
zip_filename = f"Updated_Data_Archive_{resize_suffix}_{RESIZE_MODE}.zip"
zip_path = f"/kaggle/working/{zip_filename}"
shutil.make_archive(base_name=zip_path.replace('.zip', ''), format='zip', root_dir=output_root)

print("=" * 80)
print(f"\nProcessing complete! Total execution time: {timer.elapsed()}")
print()
logging.info(f"Processing complete! Total execution time: {timer.elapsed()}")
print(f"Complete archive available at: {zip_path}")
print()
print(f"Check the '{tables_dir}' and '{visualizations_dir}' folders for detailed analysis.")
print()
print(f"Detailed error logs available at: {error_log_file}")
print()
print("=" * 80)