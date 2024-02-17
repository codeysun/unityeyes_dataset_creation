# Script for collecting data using UnityEyes
# Ensure that UnityEyes is open before running the script

import argparse
import pyautogui as pyg
import pygetwindow as gw
import random
import math
import numpy as np
import json
import os
import time
import shutil
import subprocess
import win32api
import win32con
import cv2
import PIL
from PIL import Image

UNITYEYES_PATH = "UnityEyes_Windows"
GLASSES_PATH = "glasses"
FRAMES_PER_ID = 5
IDS = 5
radians_to_degrees = 180.0 / np.pi

NUM_GLASSES = 1

SCREEN_WIDTH = 3440
SCREEN_HEIGHT = 1440

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


def rand_sign():
    return 1 if random.random() < 0.5 else -1


def sign(x):
    if x == 0:
        return 0
    if x < 0:
        return -1
    return 1


def displayMousePosition():
    while True:
        time.sleep(1)
        pyg.displayMousePosition()


def command_randomize_illumination():
    """
    randomize illumination
    """
    pyg.typewrite("l")


def command_toggle_ui():
    """
    toggle UI display
    """
    pyg.typewrite("h")


def determine_face_position_digit(face_position):
    """
    return and int that indicates the face position
    :param face_position:
    :return:
    """
    assert face_position in ["center", "left", "right", "bottom", "top"]
    face_dict = {"center": 5, "left": 4, "right": 6, "bottom": 2, "top": 8}
    return face_dict[face_position]


def process_json_list(json_list, img):
    ldmks = [eval(s) for s in json_list]
    return np.array([(x, WINDOW_HEIGHT - y, z) for (x, y, z) in ldmks])


def give_time_to_open_unity(sec=7):
    print("Data collection starting...\nReopen unity to reset img numbering")
    for s in range(sec, 0, -1):
        print(f"Start in {s} seconds, open unity in the foreground. ")
        print(f"Move face to the {FACE_POSITION}")
        time.sleep(1)


def get_looking_vec_json(json_path):
    data_file = open(json_path)
    data = json.load(data_file)
    look_vec = list(eval(data["eye_details"]["look_vec"]))
    # look_vec[1] = -look_vec[1]
    return look_vec


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.
    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1]) * radians_to_degrees  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2]) * radians_to_degrees  # phi
    if out[:, 1] > 90:
        out[:, 1] = 180 - out[:, 1]
    if out[:, 1] < -90:
        out[:, 1] = -180 - out[:, 1]
    return out


class UnityEyesDataCreator:
    def __init__(self, datatype, crop, glasses, width=360, height=180):
        self.unity_path = UNITYEYES_PATH
        self.glasses_path = GLASSES_PATH
        self.unityeyes_process = None
        self.start_unity_eyes()

        # Window settings
        x_origin, y_origin, x_width, y_height = self.reset_window_position()
        self.mouse_border = {
            "left": x_origin,
            "right": x_width,
            "bot": y_height,
            "top": y_origin,
        }
        self.window_center = [
            int(
                self.mouse_border["left"]
                + (self.mouse_border["right"] - self.mouse_border["left"]) / 2
            ),
            int(
                self.mouse_border["top"]
                + (self.mouse_border["bot"] - self.mouse_border["top"]) / 2
            ),
        ]

        # Eye gaze settings
        self.gaze_variance = 100
        self.velocity_limit = 100
        self.moment_limit = 40
        # Head pose settings
        self.head_x_range = 20  # Range of head pose variation. In degrees
        self.head_y_range = 20
        self.head_dx_px = 0
        self.head_dy_px = 0
        # self.command_randomize_head_pose(self.head_x_range, self.head_y_range)

        # Post processing
        self.crop = crop
        self.glasses = glasses

        self.debug = True
        self.datatype = datatype
        self.id_count = 0
        self.image_count = 0
        self.frame_count = 0
        self.velocity = np.asarray(
            [rand_sign() * random.randint(5, 15), rand_sign() * random.randint(5, 15)]
        )
        self.moment = np.asarray(
            [rand_sign() * random.randint(-3, -1), rand_sign() * random.randint(-3, -1)]
        )
        self.mouse_location = None
        self._center_guess = self.window_center
        self.center = self._center_guess
        self.imgs_and_json_folder = os.path.join(self.unity_path, "imgs")
        self.clean_output_folder()
        self.cutout_width = width
        self.cutout_height = height
        # self._x_correction = 0
        # self._y_correction = 0
        self.new_cutout_imgs_and_json_folder = None
        self.dataset_imgs_and_json_folder = None
        # assert not os.path.isdir(os.path.join(self.unity_path, f"imgs_{FACE_POSITION}"))

    def start_unity_eyes(self):
        try:
            print("Starting UnityEyes")
            self.unityeyes_process = subprocess.Popen(
                os.path.join(self.unity_path, "unityeyes.exe"), cwd=self.unity_path
            )
        except Exception as e:
            print(f"Error occurred while starting the application: {str(e)}")

        time.sleep(10)
        window = gw.getWindowsWithTitle("UnityEyes Configuration")[0]
        window.moveTo(0, 0)
        pyg.moveTo(350, 400)
        pyg.click()
        time.sleep(10)

    def close_unity_eyes(self):
        self.unityeyes_process.terminate()

    def reset_window_position(self):
        window = gw.getWindowsWithTitle("UnityEyes")[0]
        window.moveTo(0, 0)
        w_size = window.size
        pad_width = math.ceil((w_size[0] - WINDOW_WIDTH) / 2)
        title_bar_height = w_size[1] - WINDOW_HEIGHT - pad_width
        return (
            pad_width,
            title_bar_height,
            WINDOW_WIDTH,
            title_bar_height + WINDOW_HEIGHT,
        )

    def clean_output_folder(self):
        shutil.rmtree(self.imgs_and_json_folder)
        os.mkdir(self.imgs_and_json_folder)

    def get_last_json_path(self):
        return os.path.join(self.unity_path, "imgs", f"{self.image_count}.json")

    def get_last_img_path(self):
        return os.path.join(self.unity_path, "imgs", f"{self.image_count}.jpg")

    def command_lclick_eyes_at_rel(self, relative_distance):
        pyg.moveRel(relative_distance[0], relative_distance[1], duration=0)
        pyg.click(button="middle")
        self.mouse_location = self.mouse_location + relative_distance

    def command_click_eyes_at_loc(self, location):
        pyg.middleClick(location[0], location[1])
        self.mouse_location = location

    def command_randomize_id(self):
        """
        randomize id
        """
        self.id_count += 1
        pyg.typewrite("r")
        self.command_randomize_head_pose(self.head_x_range, self.head_y_range)

    def command_randomize_head_pose(self, x_deg_range, y_deg_range):
        """
        Randomize the head pose in uniform random distribution
        """
        # 100px movement = 20deg
        # Convert degrees to pixels
        x_px_range = x_deg_range * 5
        y_px_range = y_deg_range * 5
        direction = np.random.randint([0, 0], np.array([x_px_range, y_px_range]))
        dx = direction[0]
        dy = direction[1]

        self.command_reset_head_pose()
        self._command_move_head_pose(dx, dy)

        self.head_dx_px = dx
        self.head_dy_px = dy

    def command_reset_head_pose(self):
        self._command_move_head_pose(-self.head_dx_px, -self.head_dy_px)
        self.head_dx_px = 0
        self.head_dy_px = 0

    def _command_move_head_pose(self, dx, dy):
        x = self.window_center[0]
        y = self.window_center[1]
        pyg.moveTo(x, y)

        # 100px movement = 20deg
        # For some reason, this is the only way I can get dragging to work...
        win32api.mouse_event(
            win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0
        )  # press left click (send the first postion)
        win32api.mouse_event(
            win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0
        )  # use the MOUSEEVENTF_MOVE, send relative movement
        win32api.mouse_event(
            win32con.MOUSEEVENTF_LEFTUP, x + dx, y + dy, 0, 0
        )  # release the left click (send the second postion)

    def get_current_looking_vec(self):
        self.command_save_image()
        # time.sleep(0.02)
        captured = False
        current_looking_vec = None
        wait_time = 0.01
        while not captured:
            try:
                current_looking_vec = get_looking_vec_json(self.get_last_json_path())
                captured = True
            except:
                time.sleep(wait_time)
                if self.debug:
                    print(f"waiting {wait_time} sec for capture to complete")
                captured = False

        return current_looking_vec

    def command_save_image(self):
        """
        save image at location
        """
        pyg.typewrite("s")
        self.image_count += 1
        if self.debug:
            print(f"saving {self.image_count}.jpg")

    def find_center(self):
        """
        turn eye to face camera. the look vec is positive when the eye looks to the right and upwards
        :return:
        """
        self._center_guess = self.window_center
        abs_min_x_angle = 100
        abs_min_y_angle = 100
        eps = 0.002
        min_x_loc = self._center_guess[0]
        min_y_loc = self._center_guess[1]
        x_correction = 0
        y_correction = 0
        max_iter = 20
        print("Finding center")
        print(self._center_guess)

        # TODO: fix this. This sucks.
        for step_size in [10, 5, 2, 1, 1, 1]:
            print(f"Step size: {step_size}")
            x_looping = False
            y_looping = False
            iter = 0
            while iter < max_iter and not (x_looping and y_looping):
                self.command_click_eyes_at_loc(self._center_guess)
                current_look_vec = self.get_current_looking_vec()
                if self.debug:
                    print(
                        f"#{self.image_count}: guessing: {self._center_guess} > {[current_look_vec[0], current_look_vec[1]]} : x_looping {x_looping} y_looping {y_looping}"
                    )
                x_look_vec = current_look_vec[0]
                y_look_vec = current_look_vec[1]
                if abs(x_look_vec) <= eps and abs(y_look_vec) <= eps:
                    self.center = self._center_guess
                    print("found center for current face angle")
                    return self.center
                else:
                    # What does this do?
                    # if abs_min_x_angle + abs_min_y_angle > abs(x_look_vec) + abs(
                    #     y_look_vec
                    # ):
                    #     abs_min_x_angle = abs(x_look_vec)
                    #     min_x_loc = self._center_guess[0]
                    #     abs_min_y_angle = abs(y_look_vec)
                    #     min_y_loc = self._center_guess[1]

                    # Detect convergence
                    new_x_correction = -sign(x_look_vec)
                    x_looping = (
                        new_x_correction == 0 or new_x_correction == -x_correction
                    )
                    new_y_correction = sign(
                        y_look_vec
                    )  # Note that negative look vec is down, but positive mouse location is down
                    y_looping = (
                        new_y_correction == 0 or new_y_correction == -y_correction
                    )

                    x_correction = new_x_correction
                    y_correction = new_y_correction

                    # Clamp to mouse border
                    self._center_guess = [
                        max(
                            min(
                                self._center_guess[0] + x_correction * step_size,
                                self.mouse_border["right"],
                            ),
                            self.mouse_border["left"],
                        ),
                        max(
                            min(
                                self._center_guess[1] + y_correction * step_size,
                                self.mouse_border["bot"],
                            ),
                            self.mouse_border["top"],
                        ),
                    ]
                iter += 1

        # If we never find the center...
        self.center = self._center_guess
        self.command_click_eyes_at_loc(self.center)
        current_look_vec = self.get_current_looking_vec()
        print(
            f"Couldn't find center. closest is {current_look_vec} using {self.center} coords"
        )
        assert abs(current_look_vec[0]) < eps and abs(current_look_vec[1]) < eps
        print("Guess is close enough. proceeding")

        return self.center

    def change_id(self):
        """
        randomize id and illumination
        """

        command_randomize_illumination()
        self.command_randomize_id()

    def collect_image(self, gim, gtrans):
        """
        save_image and progress image count
        """
        self.command_save_image()
        self.process_image(self.crop, gim, gtrans)

    def process_image(self, crop=False, gim=None, gtrans=None):
        """
        catch last created image and prepare a cutout of the image with a meaningful name
        """
        current_looking_vec = get_looking_vec_json(self.get_last_json_path())
        current_looking_vec = vector_to_pitchyaw(
            np.asarray([current_looking_vec])
        )  # convert to angle
        new_name_template = "ID{}_T{}_N001_F{}_V{:4.2f}_H{:4.2f}"
        new_name = new_name_template.format(
            self.id_count,
            self.datatype,
            self.frame_count,
            current_looking_vec[0][0],
            current_looking_vec[0][1],
        )
        # take last saved image
        im = Image.open(self.get_last_img_path())
        imdata = json.load(open(self.get_last_json_path()))

        if gim is not None:
            assert gtrans is not None, "Translation must be provided for glasses"
            # Superimpose glasses on the image
            im.paste(gim, gtrans, gim)

        if crop:
            # Crop the image
            left = (WINDOW_WIDTH / 2) - self.cutout_width // 2
            right = (WINDOW_WIDTH / 2) + self.cutout_width // 2
            bottom = (WINDOW_HEIGHT / 2) + self.cutout_height // 2
            top = (WINDOW_HEIGHT / 2) - self.cutout_height // 2

            im = im.crop((left, top, right, bottom))

        # give it the name and save it
        full_new_path = os.path.join(
            self.new_cutout_imgs_and_json_folder, new_name + ".jpg"
        )
        im.save(full_new_path)
        shutil.copy(
            self.get_last_json_path(),
            os.path.join(self.new_cutout_imgs_and_json_folder, new_name + ".json"),
        )

    def randomize_eye_gaze(self):
        direction = np.random.normal(np.array(self.center), self.gaze_variance)
        # Clamp direction within mouse border
        direction = np.clip(
            direction,
            [self.mouse_border["left"], self.mouse_border["top"]],
            [self.mouse_border["right"], self.mouse_border["bot"]],
        )
        self.mouse_location = direction
        self.command_click_eyes_at_loc(self.mouse_location)
        if self.debug:
            print(
                f"#img({self.image_count}): frame:{self.frame_count}: loc={self.mouse_location}"
            )

    def progress_eye_flow(self):
        """
        progress eye flow
        """
        self.mouse_location += self.velocity + 0.5 * self.moment

        self.mouse_location[0] = max(
            min(self.mouse_location[0], self.mouse_border["right"]),
            self.mouse_border["left"],
        )
        self.mouse_location[1] = max(
            min(self.mouse_location[1], self.mouse_border["bot"]),
            self.mouse_border["top"],
        )
        self.velocity += self.moment
        self.moment += np.asarray([random.randint(-2, 2), random.randint(-2, 2)])
        if self.mouse_location[0] == self.mouse_border["right"]:
            self.velocity[0] = -10
            self.moment[0] = -6
        if self.mouse_location[0] == self.mouse_border["left"]:
            self.velocity[0] = 10
            self.moment[0] = 6
        if self.mouse_location[1] == self.mouse_border["bot"]:
            self.velocity[1] = -10
            self.moment[1] = -6
        if self.mouse_location[1] == self.mouse_border["top"]:
            self.velocity[1] = 10
            self.moment[1] = 6
        self.command_click_eyes_at_loc(self.mouse_location)

        self.velocity[0] = max(
            min(self.velocity[0], self.velocity_limit), -self.velocity_limit
        )
        self.velocity[1] = max(
            min(self.velocity[1], self.velocity_limit), -self.velocity_limit
        )
        self.moment[0] = max(min(self.moment[0], self.moment_limit), -self.moment_limit)
        self.moment[1] = max(min(self.moment[1], self.moment_limit), -self.moment_limit)
        if self.debug:
            print(
                f"#img({self.image_count}): frame:{self.frame_count}: loc={self.mouse_location}. V={self.velocity}. M={self.moment}"
            )

    def move_eyes(self):
        if self.datatype == "clip":
            self.progress_eye_flow()
        elif self.datatype == "gaussian":
            self.randomize_eye_gaze()
        else:
            raise ValueError(f"datatype {self.datatype} not supported")

    def collect_dataset(self, ids, frames_per_id, set_num):
        """
        Main function to create dataset
        :param ids:
        :param frames_per_id:
        :return:
        """

        self.new_cutout_imgs_and_json_folder = os.path.join(
            self.unity_path, f"imgs_{set_num}_cutouts"
        )
        self.dataset_imgs_and_json_folder = os.path.join(
            self.unity_path, f"imgs_{set_num}"
        )
        os.makedirs(self.new_cutout_imgs_and_json_folder, exist_ok=True)

        for id_idx in range(ids):
            self.change_id()
            self.find_center()
            self.command_click_eyes_at_loc(self.center)
            if self.debug:
                print(f"Changed ID, centering at {self.center}")
            self.frame_count = 0

            glasses_im = None
            glasses_trans = None
            if self.glasses:
                # Load the current ID data
                im = Image.open(self.get_last_img_path())
                imdata = json.load(open(self.get_last_json_path()))

                # Get random glasses template
                gid = random.randint(1, NUM_GLASSES)
                glasses_im = Image.open(os.path.join(self.glasses_path, f"g{gid}.png"))
                glasses_json = open(os.path.join(self.glasses_path, f"g{gid}.json"))
                glasses_data = json.load(glasses_json)
                # Randomly color the glasses
                gcolor = (
                    random.randint(0, 150),
                    random.randint(0, 150),
                    random.randint(0, 150),
                )
                gcolor = Image.new("RGBA", glasses_im.size, gcolor)
                glasses_im = PIL.ImageChops.multiply(glasses_im, gcolor)
                # Get the eye and glasses centroids
                gcentroid = np.array([eval(glasses_data["center"])])
                ldmks_interior_margin = process_json_list(
                    imdata["interior_margin_2d"], im
                )
                eyecentroid = np.mean(ldmks_interior_margin, axis=0)[:2]
                # Randomly scale and position
                scale = 1 + 0.5 * random.random()
                glasses_trans = np.squeeze(eyecentroid - gcentroid)
                glasses_trans += np.random.randint([0, 0], [20, 20])
                glasses_trans = tuple(int(x) for x in glasses_trans)
                # TODO: send the translation for paste command
            for frame_idx in range(frames_per_id):
                self.frame_count += 1
                self.collect_image(glasses_im, glasses_trans)
                self.move_eyes()
        print("Finished creating dataset. Moving data to storage, clearing imgs folder")
        os.rename(self.imgs_and_json_folder, self.dataset_imgs_and_json_folder)
        os.makedirs(self.imgs_and_json_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", "-c", action="store_true", default=False)
    parser.add_argument("--glasses", "-g", action="store_true", default=False)

    args = parser.parse_args()
    dataset_creator = UnityEyesDataCreator(
        datatype="gaussian", crop=args.crop, glasses=args.glasses
    )
    command_toggle_ui()
    time.sleep(0.1)
    dataset_creator.collect_dataset(ids=IDS, frames_per_id=FRAMES_PER_ID, set_num=1)
