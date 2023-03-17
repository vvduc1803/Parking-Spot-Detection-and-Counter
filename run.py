# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import cv2
import numpy as np
import torch.optim
import config

from model import Model
from utils import get_parking_spots_bboxes, empty_or_not, load_checkpoint, calc_diff

# Load model for predict parking spots empty or not empty
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, betas=(0.9, 0.999))
load_checkpoint('model_checkpoint.pt', model, optimizer, config.LR, config.DEVICE)

# Draw bounding box of parking spots
mask = cv2.imread(config.mask, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
total_spots, spots = get_parking_spots_bboxes(connected_components)

# Initialize status of all parking spots
spots_status = [None for i in range(total_spots)]

# Initialize value to compute difference between previous frame and current frame\
diffs = [None for j in spots]

# Initialize status of each line
status_line1 = {i: None for i in range(len(spots)) if spots[i][0] < 250}
status_line2 = {i: None for i in range(len(spots)) if 250 < spots[i][0] <= 500}
status_line3 = {i: None for i in range(len(spots)) if 500 < spots[i][0] <= 750}
status_line4 = {i: None for i in range(len(spots)) if 750 < spots[i][0] <= 960}
status_line5 = {i: None for i in range(len(spots)) if 960 < spots[i][0] <= 1200}
status_line6 = {i: None for i in range(len(spots)) if 1200 < spots[i][0] <= 1420}
status_line7 = {i: None for i in range(len(spots)) if 1420 < spots[i][0] <= 1650}
status_line8 = {i: None for i in range(len(spots)) if 1650 < spots[i][0]}

# Read video
cap = cv2.VideoCapture(config.video)

# Repeat compute each (step) frame
previous_frame = None
frame_nmr = 0

while cap.isOpened():
    total = total_spots
    _, frame = cap.read()

    """Compute"""
    # Compute difference between previous frame and current frame of each parking spots
    if frame_nmr % config.step == 0 and previous_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x, y, w, h = spot
            spot_crop = frame[y:y + h, x:x + w]
            diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y:y + h, x:x + w])

    # Only repeat compute for parking spots have difference about mean higher 0.4
    if frame_nmr % config.step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > config.threshold_to_repeat_compute]

        # Update status of all parking spots and parking spots in each line
        for spot_idx in arr_:
            spot = spots[spot_idx]
            x, y, w, h = spot
            spot_crop = frame[y:y + h, x:x + w]

            # Compute status
            spot_status = empty_or_not(model, spot_crop, config.DEVICE)

            # Update for all parking spots
            spots_status[spot_idx] = spot_status

            # Update for each parking spots
            if spot_idx in status_line1.keys():
                status_line1[spot_idx] = spot_status
            elif spot_idx in status_line2.keys():
                status_line2[spot_idx] = spot_status
            elif spot_idx in status_line3.keys():
                status_line3[spot_idx] = spot_status
            elif spot_idx in status_line4.keys():
                status_line4[spot_idx] = spot_status
            elif spot_idx in status_line5.keys():
                status_line5[spot_idx] = spot_status
            elif spot_idx in status_line6.keys():
                status_line6[spot_idx] = spot_status
            elif spot_idx in status_line7.keys():
                status_line7[spot_idx] = spot_status
            elif spot_idx in status_line8.keys():
                status_line8[spot_idx] = spot_status

    # Update previous frame
    if frame_nmr % config.step == 0:
        previous_frame = frame.copy()

    """Print information"""
    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        x, y, w, h = spots[spot_idx]

        # Draw bounding box for each parking spots
        if spot_status:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        else:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Compute number of available parking spots in each line
    available_spots_in_line = [len([i for i in status_line1.values() if i == False]),
                               len([i for i in status_line2.values() if i == False]),
                               len([i for i in status_line3.values() if i == False]),
                               len([i for i in status_line4.values() if i == False]),
                               len([i for i in status_line5.values() if i == False]),
                               len([i for i in status_line6.values() if i == False]),
                               len([i for i in status_line7.values() if i == False]),
                               len([i for i in status_line8.values() if i == False])]

    cv2.putText(frame, f'Line1: {available_spots_in_line[0]} / {len(status_line1.keys())}', (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Line2: {available_spots_in_line[1]} / {len(status_line2.keys())}', (325, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Line3: {available_spots_in_line[2]} / {len(status_line3.keys())}', (550, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Line4: {available_spots_in_line[3]} / {len(status_line4.keys())}', (750, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Line5: {available_spots_in_line[4]} / {len(status_line5.keys())}', (1000, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Line6: {available_spots_in_line[5]} / {len(status_line6.keys())}', (1200, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Line7: {available_spots_in_line[6]} / {len(status_line7.keys())}', (1450, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Line8: {available_spots_in_line[7]} / {len(status_line8.keys())}', (1750, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw information of available spots and recommend best line
    if len([i for i in spots_status if i == False]) < total_spots*0.1:
        cv2.putText(frame, f'Available spots: {len([i for i in spots_status if i == False])} / {total_spots}',
                    (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Full', (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif total_spots*0.1 <= len([i for i in spots_status if i == False]) < total_spots*0.3:
        cv2.putText(frame, f'Available spots: {len([i for i in spots_status if i == False])} / {total_spots}',
                    (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(frame, f'Crowded', (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(frame, f'Recommend line: {available_spots_in_line.index(max(available_spots_in_line)) + 1}',
                    (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    else:
        cv2.putText(frame, f'Available spots: {len([i for i in spots_status if i == False])} / {total_spots}', (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Clear', (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Recommend line: {available_spots_in_line.index(max(available_spots_in_line)) + 1}',
                    (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show results
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
