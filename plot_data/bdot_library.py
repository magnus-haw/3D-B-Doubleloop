# bdot_library

from numpy import matrix, array,arange,mgrid,savetxt,loadtxt,cross,isnan
from numpy import shape, sin,cos,tan,zeros,append,cumsum,std,mean,arctan2
from numpy import diff,argmax,meshgrid,dot,where,shape,pi,unique,ones,size
from numpy import nansum,sqrt
from numpy.linalg import norm
import numpy as np
import array as ar
import scipy
##from file_io_lib import readVME
from functions import smooth, r_fft
from time import strptime
from scipy.ndimage.interpolation import zoom
from scipy.special import ellipk, ellipe
import matplotlib.pyplot as plt
from Constants import mu0

def readVME(fname,cols=8192,rows=2,dtype='f'):
    '''
    Reads data from VME->IDL output files
    fname: full file path,
    cols: number of time steps,
    rows: number variables,
    dtype: desired python data type ('f'-> float)
    '''
    fin = open(fname,'rb')
    a = ar.array('f')
    a.fromfile(fin, cols*rows)
    fin.close()
    ret = []
    for i in range(0,cols*rows,cols):
        ret.append( a[i:i+cols] )
    return np.array(ret)

def get_calib_matrix(calib_fname,cluster):
    data = loadtxt(calib_fname,delimiter=',')
    m = matrix(data[cluster*3:cluster*3+3,:])
    ret = m.I
    if cluster == 3:
        ret[:,0] *= 0
        ret[0,:] *=0

    if cluster == 5:
        ret[:,1] *= 0
        ret[1,:] *=0

    return ret

def get_calib_matrix_v2(calib_fname,cluster):
    data = loadtxt(calib_fname,delimiter=',')
    return matrix(data[(cluster-1)*3:cluster*3,:])

def apply_calib(cal_matrix,B1,B2,B3):
    bmatrix = matrix([B1,B2,B3])
    return cal_matrix*bmatrix

def calibrate_18chan(B_field,calib_fname,dB=30):
    #basic calibration for Br,Bt,Bz in Tesla
    cal_B = B_field.copy()*0
    for cluster in range(0,6):
        cal_matrix = get_calib_matrix(calib_fname,cluster)
        B1 = B_field[cluster*3,:]
        B2 = B_field[cluster*3+1,:]
        B3 = B_field[cluster*3+2,:]
        cal_bmatrix = apply_calib(cal_matrix,B1,B2,B3)
        #print cal_matrix
        cal_B[cluster*3,:] = cal_bmatrix[0,:]        
        cal_B[cluster*3+1,:] = cal_bmatrix[1,:]
        cal_B[cluster*3+2,:] = cal_bmatrix[2,:]

    cal_B[9,:] = (cal_B[6,:] + cal_B[12,:])/2.
    cal_B[16,:]= (2*cal_B[13,:] - cal_B[10,:])
    return cal_B*(10.**(dB/20.))

def calibrate_chan(B_field,calib_fname,dB=30):
    #basic calibration for Br,Bt,Bz in Tesla
    cal_B = B_field.copy()*0
    for cluster in range(0,18):
        cal_matrix = get_calib_matrix_v2(calib_fname,cluster+1)
        B1 = B_field[(cluster)*3,:]
        B2 = B_field[(cluster)*3+1,:]
        B3 = B_field[(cluster)*3+2,:]
        cal_bmatrix = apply_calib(cal_matrix,B1,B2,B3)
        #print cal_matrix
        #print shape(cal_B), shape(cal_bmatrix)
        cal_B[(cluster)*3,:] = cal_bmatrix[0,:]        
        cal_B[(cluster)*3+1,:] = cal_bmatrix[1,:]
        cal_B[(cluster)*3+2,:] = cal_bmatrix[2,:]

    return cal_B*(10.**(dB/20.))

def get_meta(shot_num, folder =''):
    dbase= folder +"shots/%i/"%shot_num
    
    ###Read in metadata
    metaname = dbase+"metadata_%i.csv"%shot_num
    metadata = loadtxt(metaname,delimiter=',',dtype='str')
    metadict = dict(metadata[metadata[:,0] != "-"])
    metadict['systime'] = strptime(metadict['systime'])
    return metadict

def get_bdot(shot_num,folder=''):
    print "Shot Number: %i"%(shot_num)   
    dbase = folder +"shots/%i/"%shot_num
    
    #//////////////////#
    ### Data section ###
    #//////////////////#
    
    ###Get metadata
    metadict = get_meta(shot_num,folder)
    theta = pi*float(metadict['Bdot probe theta (deg)'])/180.
    zpos = float(metadict['Bdot probe z (cm)'])/100.

    ###Read in diagnostic data   
    name = 'Bdot'
    header_name = dbase+name+"_header_%i.meta"%shot_num
    fname = dbase+name+'_%i.dat'%(shot_num)

    ###Header data
    print "Reading header data"
    head = loadtxt(header_name,delimiter=',')
    dt_sec, ind0, rows, N, mean_bias, atten_dB, calib_factor = head[0],head[1],int(head[2]),int(head[3]),head[4],head[5],head[6]

    ### Diagnostic data
    print "Reading data"
    data = readVME(fname,cols=N,rows=rows)
    t = data[0,:]
    v = data[1:,:]

    return t,v, theta, zpos


def get_opt_trigger(shot_num,folder=''):
    ###Optical trigger data
    dbase = folder +"shots/%i/"%shot_num
    
    dt = .01 #us
    try:
        ret = readVME(dbase+"croft_optical_trigger_%i.dat"%shot_num,rows=1)
        v_trigger = ret[0]
        ind = where(v_trigger > .75)[0][0]
        t0 = (ind -.5)*dt
    except:
        print "no optical trigger"
        t0 = 18.44
        ind = 1844
    return t0,ind

def get_volt_trigger(shot_num,folder=''):
    dbase = folder +"shots/%i/"%shot_num
    name = 'HV_Probe_B'

    header_name = dbase+name+"_header_%i.meta"%shot_num
    fname = dbase+name+'_%i.dat'%(shot_num)

    ###Header data
    #print "Reading header data"
    try:
        head = loadtxt(header_name,delimiter=',')
        dt_sec, ind0, rows, N, mean_bias, atten_dB, calib_factor = head[0],head[1],int(head[2]),int(head[3]),head[4],head[5],head[6]

        dt = dt_sec*1e6 #units = us
        
        ### Diagnostic data
        #print "Reading data"
        data = readVME(fname,cols=N,rows=rows)
        t = data[0,:]
        v = data[1:,:] #- mean_bias
        v_set = v[0,10:5000]
        ind = where(v_set < -1000.)[0][0]
        t0 = (ind -.5)*dt
    except:
        print "Voltage data not found for shot %i"%shot_num
        t0 = 16.4
        ind = 1640
    return t0,ind

def get_diagnostic(name,shot_num,n=3500,trigger='voltage',dataDrive=''):
    print shot_num
    #dataDrive = "G:\data\\croft\\"
    dbase  = dataDrive+"shots/%i/"%shot_num
    #name = 'Rogowski_B'
    #name = 'HV_Probe_B'
    #name = 'Bdot'

    header_name = dbase+name+"_header_%i.meta"%shot_num
    fname = dbase+name+'_%i.dat'%(shot_num)

    ###Header data
    #print "Reading header data"
    head = loadtxt(header_name,delimiter=',')
    print head
    dt_sec, ind0, rows, N, mean_bias, atten_dB, calib_factor = head[0],head[1],int(head[2]),int(head[3]),head[4],head[5],head[6]

    ###Shot metadata
#    metadict = get_meta(shot_num)
    
    ### Diagnostic data
    #print "Reading data"
    print N, rows
    data = readVME(fname,cols=N,rows=rows)
    t = data[0,:]
    v = data[1:,:] #- mean_bias

    if trigger == "optical":
        ###Optical trigger data    
        t0,ind = get_opt_trigger(shot_num)
    else:
        t0,ind = get_volt_trigger(shot_num)

    return (t-t0)[ind-500:ind+n-500], v[:,ind-500:ind+n-500]

def get_shot(shot,time_step=-1,folder=''):

    ###Retrieve data
    t,v, theta,zpos = get_bdot(shot,folder)
    t0_optical,t0_ind = get_volt_trigger(shot,folder)
    t0 = t[t0_ind]

    ###Integrate measurements
    if time_step > 0:
        B_field = cumsum(v,axis=1)[:,t0_ind+time_step:t0_ind+time_step+1]
    else:
        B_field = cumsum(v,axis=1)[:,t0_ind:t0_ind+6000]

    ###calibrate measurements
    dbase = folder + "calibration/"
    cal_B_field = calibrate_18chan(B_field,dbase+"bdot_calibration_matrix.txt")

    return cal_B_field,(theta,zpos)

def get_shot_v2(shot,time_step=-1,folder=''):
    dbase = folder + "calibration/"

    ###Retrieve data
    t,v, theta,zpos = get_bdot(shot,folder)
    t0_optical,t0_ind = get_volt_trigger(shot,folder)
    t0 = t[t0_ind]

    ###Integrate measurements
    if time_step > 0:
        B_field = cumsum(v,axis=1)[:,t0_ind+time_step:t0_ind+time_step+1]
    else:
        B_field = cumsum(v,axis=1)[:,t0_ind:t0_ind+6000]

    ###calibrate measurements
    cal_B_field = calibrate_chan(B_field,dbase+"bdot_v2_calibration_matrix.txt")

    return cal_B_field,(theta,zpos)

def get_shot_averages(list_o_shot_nums,time_step=[0]):
    n = len(list_o_shot_nums)
    b_fields = []
    positions = []
    
    #read in b_field data
    for i in range(0,n):
        b_field,position = get_shot(list_o_shot_nums[i],time_step)
        positions.append(position)
        b_fields.append(b_field)
        #print shape(b_field)

    #define sets of diff positions
    b = np.ascontiguousarray(positions).view(np.dtype((np.void, positions.dtype.itemsize * positions.shape[1])))
    _, idx = np.unique(b, return_index=True)

    locations = positions[idx]
    print locations
    B_avgs =[]

    #loop over different sets
    for j in range(0,len(locations)):
        avg = b_field*0
        counter = 0
        
        for i in range(0,n): #avg each set
            if (positions[i]==locations[j]).all():
                avg += b_fields[i]
                counter +=1
        avg /= counter 

        B_avgs.append(avg)

    return B_avgs, locations

def get_shot_averages_v2(list_o_shot_nums,time_step=0):
    n = len(list_o_shot_nums)
    b_fields = []
    positions = []
    
    #read in b_field data
    for i in range(0,n):
        b_field,position = get_shot_v2(list_o_shot_nums[i],time_step)
        positions.append(position)
        b_fields.append(b_field)
        print shape(b_field)
        print positions
    #define sets of diff positions
    b = np.ascontiguousarray(positions).view(np.dtype((np.void, positions.dtype.itemsize * positions.shape[1])))
    _, idx = np.unique(b, return_index=True)

    locations = positions[idx]
##    locations = unique(positions)
    print locations
    B_avgs =[]

    #loop over different sets
    for j in range(0,len(locations)):
        avg = b_field*0
        counter = 0
        
        for i in range(0,n): #avg each set
            if (positions[i]==locations[j]).all():
                avg += b_fields[i]
                counter +=1
        avg /= counter 

        B_avgs.append(avg)

    return B_avgs, locations

def get_shot_averages_time_array(list_o_shot_nums,zf=1,folder=''):
    dt = zf/100.
    n = len(list_o_shot_nums)
    b_fields = []
    positions = []
    
    #read in b_field data
    for i in range(0,n):
        b_field,position = get_shot_v2(list_o_shot_nums[i],time_step=-1,folder=folder)
        positions.append(position)
        b_field = b_field[:,::zf] ##extract regular time steps
        b_fields.append(b_field)
        print shape(b_field)
    
    #define sets of diff positions
    positions = array(positions)
    b = np.ascontiguousarray(positions).view(np.dtype((np.void, positions.dtype.itemsize * positions.shape[1])))
    _, idx = np.unique(b, return_index=True)

    locations = positions[idx]
    #define sets of diff positions
##    locations = unique(positions, axis=0) # for use with numpy 1.13
    print locations
    B_avgs =[]

    #loop over different sets
    for j in range(0,len(locations)):
        print j
        avg = b_field*0
        counter = 0

        for i in range(0,n): #avg each set
            if (positions[i]==locations[j]).all():
                avg += b_fields[i]
                counter +=1
                
        avg /= counter 

        B_avgs.append(avg)

    return array(B_avgs), locations,dt

def get_plane(p0,u1,u2,n1=15,n2=15,center=False):
    """
    Defines plane in quadrilateral from origin to [0->u1,0->u2]
        input arguments are
        : origin position (fmt -> [x,y,z])
        : u1, vector 1 (fmt -> [x,y,z])
        : u2, vector 2 (fmt -> [x,y,z])
        : "n" gives minimum number of interpolation positions per axis

        outputs are
        : x,y,z; arrays of regularly spaced 3D points in quadrilateral
        : vnorm; normal unit vector, [x,y,z]
        : dA, area element size
    """

    if center:
        origin = p0 - (u1/2. + u2/2.)
    else:
        origin = p0

    x,y,z = zeros((n1,n2)),zeros((n1,n2)),zeros((n1,n2))
    for i in range(0,n1):
        for j in range(0,n2):
            position = array(origin) + i*u1/n1 + j*u2/n2
            x[i,j] = position[0]
            y[i,j] = position[1]
            z[i,j] = position[2]

    normal = cross(u1/n1,u2/n2)
    dA = norm(normal)
    vnorm = normal/dA

    return x,y,z, vnorm, dA  
    

def get_cartesian_bdot(b_cluster,rad,theta_pos,z_pos):
    '''
    Input requires 3d bdot cluster time series and cluster location
    z-postion currently simulated by time dimension

    Outputs are cartesian positions and B-field vectors
    '''

    #origin defined at electrode plane, probe axis

    # z-axis into chamber
    # y-axis vertically up
    # x-axis towards gas cylinders

    theta_pos = .5*pi + theta_pos 

    ###convert vectors to cartesian coordinates
    Br = b_cluster[0]
    Btheta = b_cluster[1]
    Bz = b_cluster[2]

    By = cos(theta_pos)*Btheta + sin(theta_pos)*Br
    Bx = cos(theta_pos)*Br + sin(theta_pos)*Btheta

    ###convert postions to cartesian coordinates
    try:
        s=len(Bz)
    except:
        s=1
    r,theta,z= ones(s)*rad,ones(s)*theta_pos,ones(s)*z_pos

    #z = cumsum(z)/2500.+.20

    x = r*cos(theta)+.39455#m
    y = r*sin(theta)-.16496#m
    z = .26-z
    return x,y,z,Bx,By,Bz


def get_all_cluster_data(b_fields,locations,zf=1):
    '''
    takes in list of locations and associated b-field measurements
    outputs arrays of positions and B-field in cartesian coordinates
    '''

    x,y,z,Bx,By,Bz = array([]),array([]),array([]),array([]),array([]),array([])
    for i in range(0,len(locations)): #loop over probe positionss
        theta_pos = locations[i][0]
        z_pos = locations[i][1]
        for j in range(0,18): #loop over radius/clusters
            if j != 5:
                b_cluster = b_fields[i][j*3:j*3+3]
                rad = (53.25-j)/100.
                if j >=10:
                    rad -= .03
                #print rad,shape(b_cluster), j
                nx,ny,nz,nBx,nBy,nBz = get_cartesian_bdot(b_cluster,rad,theta_pos,z_pos)
                x = append(x,nx)
                y = append(y,ny)
                z = append(z,nz)
                Bx= append(Bx,nBx)
                By= append(By,nBy)
                Bz= append(Bz,nBz)
    return zoom(x,zf),zoom(y,zf),zoom(z,zf),zoom(Bx,zf),zoom(By,zf),zoom(Bz,zf)


def limit_scope(rad,th,z,X,Y,Z):
    # Boolean mask for points inside region (cylindrical coords)
    R = ((X-.39455)**2. + (Y+.16496)**2.)**.5
    theta = arctan2((Y+.16496),(X-.39455))%(2*pi)
    mask = (theta<th[1])*(theta > th[0])*(Z<z[1])*(Z>z[0])*(R<rad[1])*(R>rad[0])
    return mask

def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x,y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height*np.exp(
            -(((center_x-xp)/width_x)**2+
              ((center_y-yp)/width_y)**2)/2.)
        return g
    return rotgauss

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Returns 2dgaussian((x,y),height,width_x, width_y, theta=0, offset=0)
    """

    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def moments(data):
    """Returns (height, x, y, width_x, width_y, theta=0, offset=0)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.0, 0.0

def fitgaussian2d(data,p0=None):
    """Returns (height, x, y, width_x, width_y,theta (degrees))
    the gaussian parameters of a 2D distribution found by a fit
    """
    if p0 == None:
        params = moments(data)
    else:
        params = tuple(p0)

    data[isnan(data)]=0
    err = 1./((data!=0)+.001)
    x, y = np.indices(data.shape)
    popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, (x, y), data.ravel(),
                                          p0=params, sigma= err.ravel())
    return popt, twoD_Gaussian((x,y),*popt).reshape(shape(data))
    

def evaluate_vector_cross_section(Vx,Vy,Vz,plane_params,mask=None,plot=False,color='k',scale=None):
    '''
        Generalized function for evaluating vector data in given cross section

        plane_params = (xp,yp,zp,vnorm,dA) -> cross section points, area element, and normal vector
        vx,vy,vz -> Vector values at xp,yp,zp

        returns -> vector_flux (scalar), net_force (vector), nanfraction (scalar)
    '''
    xp,yp,zp,vnorm,dA = plane_params

    ###mask extrapolated field positions
    if mask != None:
        Vx *= mask; Vy *= mask; Vz *= mask;

    ### Deal with nan values
    nanfraction = 1.*max(isnan(Vx).sum(),isnan(Vy).sum(),isnan(Vz).sum())/size(Vx)
    Vx[isnan(Vx)] = 0;Vy[isnan(Vy)] = 0;Vz[isnan(Vz)] = 0
    
    vector_flux = dot( array([nansum(Vx),nansum(Vy),nansum(Vz)]), vnorm*(dA)) # e.g calculate net current from J
    net_force = array([nansum(Vx),nansum(Vy),nansum(Vz)])*(dA**1.5)
    #print dA
    
    if plot:
        fvec = array([Vx.ravel(),Vy.ravel(),Vz.ravel()]).T
        u1 = array([ xp[-1,0]-xp[0,0],yp[-1,0]-yp[0,0],zp[-1,0]-zp[0,0]])
        u2 = array([ xp[0,-1]-xp[0,0],yp[0,-1]-yp[0,0],zp[0,-1]-zp[0,0]])
        section_shape = shape(xp)
        V = dot(fvec,u1).reshape(section_shape)
        U = dot(fvec,u2).reshape(section_shape)
        plt.quiver(U,V, color = color, scale=scale)

    return vector_flux, net_force, nanfraction


def evaluate_scalar_cross_section(V,plane_params,mask=None,
                                  plot=False,cnt_only=False,
                                  colors=['k','r','g','b']):
    '''
        Generalized function for evaluating vector data in given cross section

        plane_params = (xp,yp,zp,vnorm,dA) -> cross section points, area element, and normal vector
        V -> values at xp,yp,zp

        returns -> [Vmin,Vmax],gaussfit_params,nanfraction

        gaussfit_params = height, center_x, center_y, width_x, width_y,rotation (deg),offset
    '''
    xp,yp,zp,vnorm,dA = plane_params
    Vmax,Vmin = np.nanmax(V),np.nanmin(V)
    
    ###mask extrapolated field positions
    if mask != None:
        V *= mask
    
    ### Deal with nan values
    nanfraction = 1.*isnan(V).sum()/size(V)
    V[isnan(V)] = 0

    N = shape(xp)[0]
    p0 = None#[Vmax,N/2.,N/2.,5,5,0,0]
    fitflag = True
    try:
        p,fit = fitgaussian2d(V,p0=p0)
    except:
        p = [-99,-99,-99,-99,-99,-99,-99]
        fitflag=False
        print "Fit failed"

    if plot:
        if cnt_only == False:
            im = plt.imshow(V/1000, interpolation='gaussian',origin='lower',cmap=plt.get_cmap('hot'))
            cb = plt.colorbar(im)#,label=r"Axial Current Density,|J| (kA/m$^2$)")
##            cb.set_label(label=r'Axial Current Density ($kA/m^2$)',size=20)#,weight='bold')
        if fitflag:
            cs2 = plt.contour(V/1000, levels=[p[0]*.75/1000], hold='on', colors=colors[1],origin='lower',linewidths = 2)
            cs3 = plt.contour(V/1000, levels=[p[0]*.50/1000], hold='on', colors=colors[2],origin='lower',linewidths = 2)
            cs4 = plt.contour(V/1000, levels=[p[0]*.90/1000], hold='on', colors=colors[3],origin='lower',linewidths = 2)
            cs5 = plt.contour(fit/1000, levels=[p[0]*.75/1000], hold='on', colors=colors[1],origin='lower',linewidths = 3,linestyles='dashed')
            cs6 = plt.contour(fit/1000, levels=[p[0]*.5/1000], hold='on', colors=colors[2],origin='lower',linewidths = 3,linestyles='dashed')
            cs7 = plt.contour(fit/1000, levels=[p[0]*.90/1000], hold='on', colors=colors[3],origin='lower',linewidths = 3,linestyles='dashed')  

        #plt.axis(v)
        
        plt.draw()
    return [Vmin,Vmax],p,nanfraction
    
def loop_current(a,I,rho,z):
    alphasq = a*a + rho*rho +z*z - 2*a*rho
    betasq  = a*a + rho*rho +z*z + 2*a*rho
    beta = sqrt(betasq)
    ksq = 1.- alphasq/betasq
    C = mu0*I/pi

    Brho = (C*z/(2*alphasq*rho*beta) )* ( (a*a + rho*rho + z*z )*ellipe(ksq) - alphasq*ellipk(ksq) )
    Bz   = (C/(2*alphasq*beta) )* ( (a*a - rho*rho - z*z )*ellipe(ksq) + alphasq*ellipk(ksq) )

    return Brho,Bz
    
