import xarray
import booz_xform as bx


def getBoozerFile(file: str):
    vmeclib = xarray.open_dataset("./wout_"+file+".nc") 
    b = bx.Booz_xform()
    b.read_wout("./wout_"+file+".nc")
    b.mboz = (2*int(vmeclib["mpol"]) + 1)
    b.nboz = (2*int(vmeclib["ntor"]) + 1)
    b.run() 
    b.write_boozmn("boozer_wout_"+file+".nc")


if __name__ == "__main__":
    file_list = [
        "DIII-D", 
        "QAS", 
        "heliotron",
        # "LandremanPaul2021_QA", 
        # "LandremanPaul2021_QA_lowres", 
        # "LandremanPaul2021_QA_reactorScale_lowres", 
        # "LandremanPaul2021_QH_reactorScale_lowres" 
    ]
    for file in file_list:
        getBoozerFile(file)
