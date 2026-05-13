"""Pitch keypoint detection, polygon masking, and homography helpers.

This package wires the Roboflow ``football-field-detection-f07vi`` keypoint
pose model into the analysis pipeline. The module is only imported when
``Settings.field_detection_enabled`` is True, so optional Roboflow
dependencies do not affect the default pipeline.
"""
