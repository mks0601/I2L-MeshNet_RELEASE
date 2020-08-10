import torch

from manopth.demo import generate_random_hand


def test_generate_random_hand():
    batch_size = 3
    hand_info = generate_random_hand(batch_size=batch_size, ncomps=6)
    verts = hand_info['verts']
    joints = hand_info['joints']
    assert verts.shape == (batch_size, 778, 3)
    assert joints.shape == (batch_size, 21, 3)
