import numpy as np
import odl


def geometric_material(n, dom_width):

    U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5], [dom_width*0.5,
                          dom_width*0.5], (n, n))

    # left-big
    gt1 = odl.phantom.cuboid(U, [-0.375, -0.25], [-0.125, 0.125])
    # upper-right
    gt2 = odl.phantom.cuboid(U, [0.25, 0.25], [0.375, 0.375])
    # bottom-right-large
    gt3 = odl.phantom.cuboid(U, [0.2, -0.4], [0.3, -0.1])
    # bottom-center
    gt4 = odl.phantom.cuboid(U, [-0.05, -0.4], [0.1, -0.3])
    # hole inside block small
    gt5 = odl.phantom.cuboid(U, [-0.3, -0.1], [-0.2, 0.05])
    # hole inside block small
    gt6 = odl.phantom.cuboid(U, [-0.3, -0.2], [-0.275, -0.175])

    materialA = gt1 + gt2 + gt3 + gt4 - gt5 - gt6

    # vertical lines
    gt7 = odl.phantom.cuboid(U, [0, -0.25], [0.005, 0.1])  # left
    gt8 = odl.phantom.cuboid(U, [0.015, -0.25], [0.02, 0.1])  # right
    gt9 = odl.phantom.cuboid(U, [0.07, -0.25], [0.075, 0.1])  # right2

    # horizontal lines
    gt10 = odl.phantom.cuboid(U, [-0.25, 0.245], [0.1, 0.255])  # inf
    gt11 = odl.phantom.cuboid(U, [-0.25, 0.275], [0.1, 0.275])  # sup
    gt12 = odl.phantom.cuboid(U, [-0.075, 0.35], [0.1, 0.36])

    materialB = gt7 + gt8 + gt9 + gt10 + gt11 + gt12

    gt13 = odl.phantom.cuboid(U, [-0.05, 0.4], [0.005, 0.41])  # sup
    gt14 = odl.phantom.cuboid(U, [0.125, -0.125], [0.135, -0.115])  # middle
    gt15 = odl.phantom.cuboid(U, [0.35, -0.005], [0.365, 0.005])  # middle-rig
    gt16 = odl.phantom.cuboid(U, [-0.2, -0.3], [-0.185, -0.285])  # inf left
    materialC = gt13 + gt14 + gt15 + gt16

    Ut = np.zeros((n, n, 3))
    Ut[:, :, 0] = materialA.asarray()
    Ut[:, :, 1] = materialB.asarray()
    Ut[:, :, 2] = materialC.asarray()
    return Ut


def geometric_phantom_sinfo(n, dom_width):
    U = odl.uniform_discr([-dom_width*0.5, -dom_width*0.5], [dom_width*0.5,
                          dom_width*0.5], (n, n))

    # left-big
    gt1 = odl.phantom.cuboid(U, [-0.375, -0.25], [-0.125, 0.125])
    # upper-right
    gt2 = odl.phantom.cuboid(U, [0.25, 0.25], [0.375, 0.375])
    # bottom-right-large
    gt3 = odl.phantom.cuboid(U, [0.2, -0.4], [0.3, -0.1])
    # line
    gt11 = odl.phantom.cuboid(U, [-0.25, 0.275], [0.1, 0.275])

    return gt1 + gt2 + gt3 + gt11
