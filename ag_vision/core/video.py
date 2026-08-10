import numpy as np
from pymediainfo import MediaInfo
import os
import subprocess
import cv2

from open_aglabs.video.models import AgVideoModel

from ag_vision.data_io import local_io, databricks_io
from ag_vision.constants import paths
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)  # Use __name__ to get the module's name
import imageio_ffmpeg

ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()


class AgVideo:
    def __init__(self, platform: str = None, cloud_bucket: str = None, video_key: str = None,
                 metadata_key: str = None, video: np.ndarray or None = None, metadata: AgVideoModel or None = None):
        """

        """
        self.platform = platform
        self.cloud_bucket = cloud_bucket
        self.video_key = video_key
        self.metadata_key = metadata_key
        self.video = video
        self.metadata = metadata
        self.exif = None

    def __repr__(self):
        return f"AgVideo(key={self.video_key}, metadata={self.metadata})"

    def generate_metadata_key_from_img_key(self):
        assert self.video_key is not None, "Video key needs to be set."
        self.metadata_key = paths.generate_metadata_path_from_file_name(data_path=self.video_key)

    def load_video(self):
        """
        Loads video data into self.video as a cv2.VideoCapture object.
        Supports loading from local disk, AWS S3, or Databricks depending on self.platform.

        Raises:
            AssertionError: If video_key or platform is not set.
            ValueError: If the platform is not supported or the video fails to open.
        """
        assert self.video_key is not None, "video_key must be set before loading."
        assert self.platform is not None, "platform must be set before loading."

        if self.platform == "local":
            if not os.path.exists(self.video_key):
                raise FileNotFoundError(f"Video file not found at: {self.video_key}")
            cap = local_io.read_video(self.video_key)

        elif self.platform == "db":
            assert self.cloud_bucket is not None, "cloud_bucket must be set for Databricks platform."

            cap = databricks_io.read_video_from_databricks(self.video_key)

        else:
            raise ValueError(f"Unsupported platform: '{self.platform}'. Must be one of: 'local', 'aws', 'db'.")

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_key}")

        logger.info(f"Video loaded successfully: {self.video_key}")
        self.video = cap

    def save_video(self, save_key: str = None, fourcc: str = 'mp4v', fps: float = None):
        """
        Saves the loaded video (cv2.VideoCapture) frame-by-frame to a local path or Databricks.

        Args:
            save_key (str): The destination path to save the video. Defaults to self.video_key if not provided.
            fourcc (str): The 4-character codec code for the output video. Defaults to 'mp4v'.
            fps (float): Frames per second for the output video. If None, uses the source video's FPS.

        Raises:
            AssertionError: If self.video is not set or is not opened.
            ValueError: If the platform is not supported.
        """
        assert self.video is not None and self.video.isOpened(), "No video loaded. Call load_video() first."
        assert self.platform is not None, "platform must be set before saving."

        dst_key = save_key or self.video_key
        assert dst_key is not None, "A save path must be provided either via save_key or self.video_key."

        if self.platform == 'local':
            try:
                local_io.save_video(self.video,
                                    save_path=dst_key,
                                    fourcc=fourcc,
                                    fps=fps)
            except Exception as e:
                logger.warning(f"Exception occurred while saving video locally: {e}")

        elif self.platform == 'db':
            try:
                databricks_io.save_video_to_databricks(video=self.video,
                                                       file_name=dst_key,
                                                       fourcc=fourcc,
                                                       fps=fps)
            except Exception as e:
                logger.warning(f"Exception occurred while saving video to Databricks: {e}")

        else:
            raise ValueError(f"Unsupported platform: '{self.platform}'. Must be one of: 'local', 'db'.")

    def convert_mov_to_mp4(self, output_f_name: str) -> bool:
        """
        Converts the source video (e.g. .mov) to an MP4 file using ffmpeg.

        Args:
            output_f_name (str): The output file path for the converted MP4.

        Returns:
            bool: True if conversion succeeded, False otherwise.

        Raises:
            AssertionError: If video_key is not set.
        """
        assert self.video_key is not None, "video_key must be set before converting."

        result = subprocess.run(
            [ffmpeg_bin, '-i', self.video_key, '-c:v', 'copy', '-c:a', 'copy', output_f_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            logger.warning(f"ffmpeg conversion failed: {result.stderr.decode()}")
            return False

        logger.info(f"Video converted successfully to: {output_f_name}")
        return True

    def extract_metadata_from_exif(self):
        """
        Extracts metadata from the exif file.
        """
        assert self.metadata is not None, "Metadata needs to be intialized."

        try:
            media_info = MediaInfo.parse(self.video_key)

            general = next((t.to_data() for t in media_info.tracks if t.track_type == 'General'), {})
            video_track = next((t.to_data() for t in media_info.tracks if t.track_type == 'Video'), {})

            self.metadata.acquisition_properties.date = (
                    general.get("encoded_date")
                    or general.get("tagged_date")
                    or general.get("file_creation_date")
            )

            # 2. Fall back to standard container keys for non-Apple devices
            self.metadata.camera_properties.make = (
                    general.get("comapplequicktimemake")
                    or general.get("make")
                    or 'Unknown'
            )
            self.metadata.camera_properties.model = (
                    general.get("comapplequicktimemodel")
                    or general.get("model")
                    or 'Unknown'
            )
            self.metadata.location_properties.raw_string = (
                    general.get("comapplequicktimelocationiso6709")
                    or general.get("location")
                    or general.get("xyz")
            )

            # 3. Read video track specs safely
            self.metadata.video_quality.height = video_track.get("height")
            self.metadata.video_quality.width = video_track.get("width")
            self.metadata.video_quality.frame_rate = video_track.get("frame_rate")
            self.metadata.video_quality.rotation = video_track.get("rotation", 0)
            self.metadata.video_quality.frames = video_track.get("frame_count") or video_track.get("count")

        except Exception as e:
            print(e)

    def load_metadata_from_dict(self, metadata_dict: dict):
        try:
            self.metadata = AgVideoModel(**metadata_dict)
        except Exception as e:
            self.metadata = None
            logger.warning(f'Exception occurred while reading metadata from DBFS: {e}')

    def save_frames_at_interval(self, interval_seconds: float, output_dir: str, resolution='jpg'):
        """
        Saves a JPG frame every specified interval seconds by checking the video's FPS.

        Args:
            interval_seconds (float): The time interval (in seconds) between saving frames.
            output_dir (str): The directory path where images will be saved.
            resolution (str): The image format ('jpg', 'png'). Defaults to 'jpg'.

        Returns:
            int: The number of frames successfully saved.
        """
        if self.video is None or not self.video.isOpened():
            raise AssertionError("Cannot save frames; video object is not loaded or opened.")

        # Calculate FPS and the minimum frame count needed for this interval
        fps = self.video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.error("Could not determine Frames Per Second (FPS). Cannot calculate reliable time intervals.")
            return 0

        frames_to_skip = int(interval_seconds * fps)
        count = 0
        saved_files = []

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            f"Starting frame extraction: Saving one image every {interval_seconds} seconds ({frames_to_skip} frames).")

        while True:
            ret, frame = self.video.read()
            if not ret:
                break  # End of video stream

            count += 1

            # Check if enough frames have passed since the start (or last saved frame)
            # We use modulo division for efficiency, ensuring we only save every Nth frame.
            if (count - 1) % frames_to_skip == 0:
                filename = os.path.join(output_dir,
                                        f"{str(uuid4())}.{resolution}")

                # Save the frame using cv2.imwrite
                success = cv2.imwrite(filename, frame)

                if success:
                    saved_files.append(filename)
                    logger.debug(f"Saved frame {count} to {filename}")
                else:
                    logger.warning(f"Failed to save frame {count}.")

        logger.info(f"Finished saving frames. Total saved files: {len(saved_files)}")
        return len(saved_files)

    def read_metadata(self):
        assert self.metadata_key is not None, "metadata key needs to be set."

        if self.platform == 'db':
            if os.path.exists(self.metadata_key):
                try:
                    m_data = databricks_io.read_json_from_databricks(file_name=self.metadata_key)
                    self.metadata = AgVideoModel(**m_data)
                except Exception as e:
                    self.metadata = None
                    logger.warning(f'Exception occurred while reading metadata from DBFS: {e}')

        elif self.platform == 'local':
            try:
                m_data = local_io.read_json(file_path=self.metadata_key)
                self.metadata = AgVideoModel(**m_data)
            except Exception as e:
                self.metadata = None
                logger.warning(f'Exception occurred while reading metadata from local: {e}')
        else:
            logger.warning(f'The cloud platform need to be local or db')

    def save_metadata(self):
        assert self.metadata is not None, "Metadata is none, will not save."
        assert self.metadata_key is not None, "Metadata key needs to be set."

        if self.metadata is not None and self.metadata_key is not None:
            if self.platform == 'db':
                try:
                    databricks_io.save_json_to_databricks(data=self.metadata.model_dump(),
                                                          file_name=self.metadata_key)

                except Exception as e:
                    logger.warning(f'Exception occurred while saving metadata to DB: {e}')

            elif self.platform == 'local':
                try:
                    local_io.save_json(data=self.metadata.model_dump(),
                                       file_path=self.metadata_key)
                except Exception as e:
                    logger.warning(f'Exception occurred while saving metadata to local: {e}')

        else:
            logger.warning(
                f'One of the following attributes is None: metadata: {self.metadata}, cloud_metadata_key: {self.metadata_key}')

    def initialize_metadata(self, device: str = None, img_type: str = None):
        if self.video_key:
            metadata_dict = {
                "path": self.video_key,
                "id": self.video_key.split('/')[-1].split('.')[0],
                "device": device,
                "type": img_type,
                "protocol_properties": {},
                "camera_properties": {},
                "location_properties": {},
                "acquisition_properties": {},
                "image_quality": {}
            }
            self.metadata = AgVideoModel(**metadata_dict)
        else:
            logger.warning(f'The image key is None and is needed to initialize the metadata.')