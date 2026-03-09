"""
Flask Application for AI-Based Abnormal Behavior Detection from Surveillance Cameras
AI Vision - Dashboard System
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
import logging
from werkzeug.utils import secure_filename
import uuid
import numpy as np
from collections import defaultdict
from detect_track_action import process_video_with_tracking

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALERTS_FOLDER = os.path.join('static', 'outputs')
CHECKPOINTS_FOLDER = 'checkpoints'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, ALERTS_FOLDER, CHECKPOINTS_FOLDER]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# In-memory storage for processing status
processing_jobs = {}
alert_statistics = defaultdict(lambda: {'normal': 0, 'fire': 0, 'fighting': 0, 'fall': 0})
building_alerts = defaultdict(lambda: defaultdict(int))


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_all_alerts():
    """Load all alerts from JSON files"""
    alerts = []
    if os.path.exists(ALERTS_FOLDER):
        for file in os.listdir(ALERTS_FOLDER):
            if file.startswith('alerts_') and file.endswith('.json'):
                try:
                    with open(os.path.join(ALERTS_FOLDER, file), 'r') as f:
                        file_alerts = json.load(f)
                        # Ensure file_alerts is a list
                        if isinstance(file_alerts, list):
                            # Filter out non-dict items
                            valid_alerts = [alert for alert in file_alerts if isinstance(alert, dict)]
                            alerts.extend(valid_alerts)
                        elif isinstance(file_alerts, dict):
                            # Single alert as dict
                            alerts.append(file_alerts)
                except Exception as e:
                    logger.error(f"Error loading alerts from {file}: {e}")
    
    # Sort with proper timestamp handling (convert to float)
    def get_timestamp(alert):
        ts = alert.get('timestamp', 0)
        if isinstance(ts, str):
            try:
                # Try parsing ISO format
                from datetime import datetime
                return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except:
                return 0
        return float(ts) if ts else 0
    
    return sorted(alerts, key=get_timestamp, reverse=True)


def get_statistics():
    """Calculate alert statistics"""
    all_alerts = get_all_alerts()
    stats = {
        'total_alerts': len(all_alerts),
        'by_type': {'normal': 0, 'fire': 0, 'fighting': 0, 'fall': 0, 'arson': 0, 'dumping': 0},
        'by_building': {},  # Regular dict instead of defaultdict
        'recent': all_alerts[:10]
    }
    
    for alert in all_alerts:
        # Ensure alert is a dict
        if not isinstance(alert, dict):
            continue
            
        alert_type = alert.get('type', 'normal').lower()
        if alert_type in stats['by_type']:
            stats['by_type'][alert_type] += 1
        
        building = alert.get('building', 'Unknown')
        region = alert.get('region', alert.get('camera_region', 'Unknown'))
        if building not in stats['by_building']:
            stats['by_building'][building] = {'accident': 0, 'arson': 0, 'dumping': 0}
    
    return stats


def process_video_async(video_path, job_id):
    """Process video in background thread with YOLOv11 + SlowFast/ActionFormer"""
    logger.info(f"[JOB {job_id}] Starting video processing thread")
    try:
        processing_jobs[job_id]['status'] = 'processing'
        processing_jobs[job_id]['current_frame'] = None
        logger.info(f"[JOB {job_id}] Status set to 'processing'")
        
        # Get checkpoint path for SlowFast+Pose model
        checkpoint = os.path.join(CHECKPOINTS_FOLDER, 'best_model_pose.pth')
        logger.info(f"[JOB {job_id}] Checkpoint path: {checkpoint}")
        logger.info(f"[JOB {job_id}] Checkpoint exists: {os.path.exists(checkpoint)}")
        
        # YOLOv11 custom model
        yolo_model = 'checkpoints/customyolov11m.pt'
        logger.info(f"[JOB {job_id}] YOLO model path: {yolo_model}")
        logger.info(f"[JOB {job_id}] YOLO model exists: {os.path.exists(yolo_model)}")
        
        # Always generate output video with both models' results
        output_video = os.path.join(OUTPUT_FOLDER, f'processed_{job_id}.mp4')
        logger.info(f"[JOB {job_id}] Output video path: {output_video}")
        
        # Process video with both models
        def progress_callback(progress):
            processing_jobs[job_id]['progress'] = progress
            logger.info(f"[JOB {job_id}] Progress: {progress:.1f}%")
        
        # Frame callback for real-time streaming
        def frame_callback(frame_rgb, frame_idx, total_frames):
            import base64
            import cv2
            # Resize frame for streaming (smaller size for faster transfer)
            frame_small = cv2.resize(frame_rgb, (640, 480))
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
            # Convert to base64
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            processing_jobs[job_id]['current_frame'] = {
                'frame': frame_b64,
                'frame_idx': frame_idx,
                'total_frames': total_frames,
                'progress': (frame_idx / total_frames * 100) if total_frames > 0 else 0
            }
        
        logger.info(f"[JOB {job_id}] Calling process_video_with_tracking with SlowFast+Pose model...")
        _, alerts = process_video_with_tracking(
            video_path,
            checkpoint,
            yolo_model,
            output_video,
            conf_threshold=0.5,
            progress_callback=progress_callback,
            frame_callback=frame_callback
        )
        logger.info(f"[JOB {job_id}] SlowFast+Pose processing completed. Alerts: {len(alerts)}")
        
        # Add metadata to alerts
        for alert in alerts:
            alert['upload_time'] = datetime.now().isoformat()
            alert['video_id'] = job_id
        
        # Save alerts to JSON file
        alerts_file = os.path.join(
            ALERTS_FOLDER,
            f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        logger.info(f"[JOB {job_id}] Alerts saved to: {alerts_file}")
        
        processing_jobs[job_id]['status'] = 'completed'
        processing_jobs[job_id]['alerts'] = alerts
        processing_jobs[job_id]['output_video'] = output_video
        processing_jobs[job_id]['alert_count'] = len(alerts)
        
        logger.info(f"[JOB {job_id}] Video processed successfully with both models. Alerts: {len(alerts)}")
        
    except Exception as e:
        logger.error(f"[JOB {job_id}] ERROR processing video: {e}", exc_info=True)
        processing_jobs[job_id]['status'] = 'failed'
        processing_jobs[job_id]['error'] = str(e)
        processing_jobs[job_id]['error_details'] = str(e)


# ==================== ROUTES ====================

@app.route('/')
def dashboard():
    """Main dashboard page - Single Page Application"""
    return render_template('dashboard.html')


@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Upload and process video with YOLOv11 + SlowFast/ActionFormer"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    job_id = str(uuid.uuid4())
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}_{filename}')
    file.save(filepath)
    
    # Initialize job
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'filename': filename,
        'upload_time': datetime.now().isoformat(),
        'filepath': filepath
    }
    
    # Start processing in background with both models
    thread = threading.Thread(
        target=process_video_async,
        args=(filepath, job_id)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': 'Video uploaded successfully. Processing with YOLOv11 + SlowFast+Pose (skeleton) model.',
        'status': 'queued'
    }), 202


@app.route('/api/processing-status/<job_id>', methods=['GET'])
def processing_status(job_id):
    """Get video processing status with current frame"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    response = {
        'job_id': job_id,
        'status': job['status'],
        'progress': job.get('progress', 0),
        'alert_count': job.get('alert_count', 0),
        'error': job.get('error', None)
    }
    
    # Add current frame if available (for real-time display)
    if 'current_frame' in job and job['current_frame'] is not None:
        response['current_frame'] = job['current_frame']
    
    return jsonify(response)


@app.route('/api/download-video/<job_id>', methods=['GET'])
def download_video(job_id):
    """Download processed video"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Video not ready'}), 400
    
    if 'output_video' in job and os.path.exists(job['output_video']):
        return send_file(job['output_video'], as_attachment=True)
    
    return jsonify({'error': 'Output video not found'}), 404


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get filtered alerts"""
    try:
        job_id = request.args.get('job_id', None)
        building = request.args.get('building', None)
        alert_type = request.args.get('type', None)
        date_from = request.args.get('date_from', None)
        date_to = request.args.get('date_to', None)
        limit = request.args.get('limit', 100, type=int)
        
        # If job_id specified, return alerts from processing job
        if job_id and job_id in processing_jobs:
            job = processing_jobs[job_id]
            if 'alerts' in job:
                return jsonify(job['alerts'])
            return jsonify([])
        
        # Otherwise return all historical alerts
        alerts = get_all_alerts()
        
        # Apply filters
        if building:
            alerts = [a for a in alerts if a.get('building', '').lower() == building.lower()]
        if alert_type:
            alerts = [a for a in alerts if a.get('type', '').lower() == alert_type.lower()]
        
        # Limit results
        alerts = alerts[:limit]
        
        return jsonify(alerts)
    except Exception as e:
        logger.error(f"Error in get_alerts: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_stats():
    """Get statistics data"""
    try:
        stats = get_statistics()
        
        # Format for charts
        alert_by_type = stats['by_type']
        
        # Generate monthly data for charts
        now = datetime.now()
        monthly_data = []
        for i in range(12):
            month_start = now - timedelta(days=30*i)
            monthly_data.append({
                'month': month_start.strftime('%B'),
                'alerts': np.random.randint(0, 10)
            })
        
        return jsonify({
            'total_alerts': stats['total_alerts'],
            'by_type': alert_by_type,
            'monthly': monthly_data,
            'recent': stats['recent']
        })
    except Exception as e:
        logger.error(f"Error in get_stats: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/summary', methods=['GET'])
def alerts_summary():
    """Get alerts summary by building"""
    try:
        alerts = get_all_alerts()
        
        summary = {}  # Regular dict instead of defaultdict
        for alert in alerts:
            # Ensure alert is a dict
            if not isinstance(alert, dict):
                continue
                
            building = alert.get('building', 'Unknown')
            alert_type = alert.get('type', 'normal').lower()
            
            if building not in summary:
                summary[building] = {'accident': 0, 'arson': 0, 'dumping': 0}
            
            if alert_type in summary[building]:
                summary[building][alert_type] += 1
        
        # Convert to list with building names
        result = []
        for building, counts in summary.items():
            result.append({
                'building': building,
                **counts
            })
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in alerts_summary: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all processing jobs"""
    jobs_list = []
    for job_id, job in processing_jobs.items():
        jobs_list.append({
            'job_id': job_id,
            'filename': job.get('filename', ''),
            'status': job['status'],
            'progress': job.get('progress', 0),
            'building': job.get('building', ''),
            'upload_time': job.get('upload_time', ''),
            'alert_count': job.get('alert_count', 0)
        })
    
    return jsonify(sorted(jobs_list, key=lambda x: x['upload_time'], reverse=True))


@app.route('/api/uploads', methods=['GET'])
def list_uploads():
    """List uploaded files"""
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(filepath):
                files.append({
                    'name': file,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
    
    return jsonify(sorted(files, key=lambda x: x['modified'], reverse=True))


@app.route('/api/delete-alert/<int:alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """Delete alert (placeholder)"""
    # In a real app, this would delete from database
    return jsonify({'message': 'Alert deleted'}), 200


@app.route('/api/files', methods=['GET'])
def list_files():
    """List all files (uploads and outputs)"""
    files = []
    
    # List uploaded files
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(filepath):
                files.append({
                    'name': file,
                    'size': os.path.getsize(filepath),
                    'type': 'upload',
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
    
    # List output files
    if os.path.exists(OUTPUT_FOLDER):
        for file in os.listdir(OUTPUT_FOLDER):
            filepath = os.path.join(OUTPUT_FOLDER, file)
            if os.path.isfile(filepath):
                files.append({
                    'name': file,
                    'size': os.path.getsize(filepath),
                    'type': 'output',
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
    
    return jsonify(sorted(files, key=lambda x: x['modified'], reverse=True))


@app.route('/api/delete-file/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete a file"""
    # Sanitize filename
    filename = secure_filename(filename)
    
    # Try to delete from uploads
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(upload_path):
        try:
            os.remove(upload_path)
            return jsonify({'success': True, 'message': 'File deleted'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Try to delete from outputs
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            return jsonify({'success': True, 'message': 'File deleted'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File not found'}), 404


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting AI Vision - AI-Based Abnormal Behavior Detection System")
    # Disable debug mode to prevent auto-restart and losing processing jobs
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
