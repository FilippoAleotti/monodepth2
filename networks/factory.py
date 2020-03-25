from networks.monodepth2 import resnet
from networks.pydnet import pydnet
from networks.fastdepth import mobilenet
from networks.omeganet import dsnet
from networks import pose

ARCHITECTURE_FACTORY = {
    'pydnet': pydnet,
    'resnet': resnet,
    'mobilenet': mobilenet,
    'dsnet': dsnet
}

def get_encoder(architecture):
    AVAILABLE_ARCHITECTURES = ARCHITECTURE_FACTORY.keys()
    assert(architecture in AVAILABLE_ARCHITECTURES)
    return ARCHITECTURE_FACTORY[architecture].Encoder

def get_decoder(architecture):
    AVAILABLE_ARCHITECTURES = ARCHITECTURE_FACTORY.keys()
    assert(architecture in AVAILABLE_ARCHITECTURES)
    return ARCHITECTURE_FACTORY[architecture].Decoder

def get_pose_decoder():
    return pose.Decoder
