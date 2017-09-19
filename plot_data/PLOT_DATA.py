from numpy import array,append,arange,savetxt,isnan,shape
from scipy import interpolate
import cPickle as pickle

from bdot_library import limit_scope,get_all_cluster_data,get_shot_averages_time_array
from functions import curl,interpolate3D, get_rect_grid
from Constants import mu0

plot3D = True
use_raw_data = False

### Limits on data region for interpolation purposes
rlim = [.3305,.527]; tlim= [2.4,3.4]; zlim=[.0,.3]
### Interpolation (min #cells/dimension)
inter_num = 10
### Data files with 5 shot averages 
load_data = ["B_avgs_shots1150-1684.pickle","B_avgs_shots1695-1834.pickle"]

### Time step, 100=1us after breakdown
tstep = 300
print "tstep = %i"%tstep

########################################################
########### Read in calibrated B-field data ############
########################################################
print "loading pickled data..."
B_avgs, locations,dt= pickle.load( open(load_data[0], "rb" ) )
B_avgs1, locations1,dt= pickle.load( open(load_data[1], "rb" ) )
B_avgs = append(B_avgs,B_avgs1,axis=0)
locations= append(locations,locations1,axis=0)
x,y,z,Bx,By,Bz = get_all_cluster_data(B_avgs[:,:,tstep/int(dt*100)],locations,zf=1)        

if use_raw_data:
    import os
    dbase = os.path.dirname(os.path.realpath(__file__)).strip("plot_data") + "raw_data/"
    ### Get b-field arrays at each probe position 1150-1684, 1695-1834
    shots = append(arange(1150,1685),arange(1695,1835))
    B_avgs, locations,dt = get_shot_averages_time_array(shots,zf=50,folder=dbase)

    ### Convert to cartesian
    print "converting to cartesian coords..."
    x,y,z,Bx,By,Bz = get_all_cluster_data(B_avgs[:,:,tstep/int(dt*100)],locations,zf=1)
    
########################################################
### Calculate interpolated B-field, J field, and JxB ###
########################################################
print "interpolating B-field..."
X,Y,Z,dx,dy,dz = get_rect_grid(x,y,z,inter_num)
Bx_inter,By_inter,Bz_inter = interpolate3D(x,y,z,Bx,By,Bz,X,Y,Z)

##savetxt('Bx_t=%04d.txt.gz'%tstep,Bx_inter.ravel(),delimiter=',')
##savetxt('By_t=%04d.txt.gz'%tstep,By_inter.ravel(),delimiter=',')
##savetxt('Bz_t=%04d.txt.gz'%tstep,Bz_inter.ravel(),delimiter=',')
##savetxt('X.txt.gz',X.ravel(),delimiter=',')
##savetxt('Y.txt.gz',Y.ravel(),delimiter=',')
##savetxt('Z.txt.gz',Z.ravel(),delimiter=',')

### calculate current density
print "calculating current density"
Jx,Jy,Jz = curl(Bx_inter/mu0,By_inter/mu0,Bz_inter/mu0,dx,dy,dz)

### calculate JxB force
jcb_x = Jy*Bz_inter - Jz*By_inter
jcb_y = Jz*Bx_inter - Jx*Bz_inter
jcb_z = Jx*By_inter - Jy*Bx_inter

### mask points outside data region
mask = limit_scope(rlim,tlim,zlim,X,Y,Z)
datalist =[Jx,Jy,Jz,Bx_inter,By_inter,Bz_inter,jcb_x,jcb_y,jcb_z]
for data in datalist:
    data *=mask
    data[isnan(data)]=0


########################################################
################# 3D Plotting Section ##################
########################################################
if plot3D:
    from mayavi import mlab
    from CroFT_electrodes import plot_all
    ###calculate Bmag and Jmag
    print "calculating Bmag, Jmag"
    Bsrc = mlab.pipeline.vector_field(X,Y,Z,Bx_inter,By_inter,Bz_inter)
    Jsrc = mlab.pipeline.vector_field(X,Y,Z,Jx,Jy,Jz)
    Fsrc = mlab.pipeline.vector_field(X,Y,Z,jcb_x,jcb_y,jcb_z)

    Bmag = mlab.pipeline.extract_vector_norm(Bsrc)
    Jmag = mlab.pipeline.extract_vector_norm(Jsrc)
    Fmag = mlab.pipeline.extract_vector_norm(Fsrc)

    print "plotting..."
    loopA_footpoints, loopB_footpoints= plot_all()

##    vectors0 = mlab.pipeline.vectors(Bsrc,
##                                scale_factor=.06,
##                                colormap='jet')
    vectors = mlab.pipeline.vectors(Jsrc,
                                scale_factor=.06,
                                colormap='hot')
    vectors.glyph.glyph_source.glyph_source.filled = True

    lines = mlab.pipeline.streamline(Bmag,seedtype='plane',
                                seed_visible=False,
                                seed_scale=0.5,
                                seed_resolution=10,
                                integration_direction = 'both',
                                )

    iso = mlab.pipeline.iso_surface(Jmag,opacity=.3)
    iso.actor.mapper.interpolate_scalars_before_mapping = True
    iso.module_manager.scalar_lut_manager.lut_mode ='autumn'
    bg_color = (1, 1, 0.94118)
    iso.scene.background = bg_color
    mlab.draw()

    mlab.view(focalpoint=array([-0.1, -0.05, 0.09]),elevation = 85.,azimuth=340.,distance =.83)
##    mlab.savefig('JMAGN_tstep=%i.png'%tstep)
    mlab.show()


