import xarray 
import booz_xform as bx

file = "vacuum" 
vmeclib = xarray.open_dataset("./wout_"+file+".nc") 
b = bx.Booz_xform() 
b.read_wout("./wout_"+file+".nc") 
b.mboz = (2*int(vmeclib["mpol"]) + 1)
b.nboz = (2*int(vmeclib["ntor"]) + 1)
b.run()
b.write_boozmn("boozer_wout_"+file+".nc")