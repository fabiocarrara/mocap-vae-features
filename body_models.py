from collections import namedtuple

BodyModel = namedtuple('BodyModel', [
    'num_dimensions',
    'num_joints',
    'edges'
])

body_models = {
    'hdm05': BodyModel(
        num_dimensions=3,
        num_joints=31,
        edges=(
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), # leg
            (0, 6), (6, 7), (7, 8), (8, 9), (9, 10), # leg
            (0, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), # torso + head
            (13, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (20, 23), # hand
            (13, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (27, 30) # hand
        ),
    ),

    'pku-mmd': BodyModel(
        num_dimensions=3,
        num_joints=25,
        edges=(
            (0, 1), (0, 12), (0, 16), 
            (1, 20), (12, 13), (16, 17), 
            (20, 2), (20, 4), (20, 8), (13, 14), (17, 18), 
            (2, 3), (4, 5), (8, 9), (14, 15), (18, 19), 
            (5, 6), (9, 10), 
            (6, 7), (6, 22), (10, 11), (10, 24), 
            (7, 21), (11, 23)
        ),
    ),
}

def get_by_name(name):
    assert name in body_models.keys(), f"Unknown body model: {name}"
    return body_models[name]

