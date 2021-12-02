
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter
from scipy.sparse.linalg import lsmr as iterative_solver
from itertools import combinations

from matplotlib import pyplot as plt
from scipy import ndimage

from lib.fourier_centeral import ff , iff
from scipy.signal import find_peaks
import cv2

from lib.pinhole_generator import pinholes
from lib.fourier_centeral import ff , iff

from skimage.restoration import unwrap_phase




def read_file(filepath):

    images_array = []

    try:
        
        with open(filepath,"rb") as f:

            image = np.load(f)
            images_array.append(image)

        return np.array(images_array)
            
    except Exception as e:
        print(e)
        return None


def get_aoi_locations(images_array, aoi_size, threshold=0.05):

    maxval = images_array.max()
    pixel_threshold = maxval*threshold
    images_array[images_array<pixel_threshold] = 0

    # find highest regional maxima in aoi_sized window
    # this relies on there being exactly one true maximum in any aoi-sized window
    # if two pixels in summed_array within a window are exactly the same value, they will both be treated as aoi locations (this almost never happens)

    maxfiltered_image=maximum_filter(images_array, size=aoi_size)
    maxfiltered_image_reversed = maximum_filter(images_array[::-1,::-1], size=aoi_size)
    local_maxima = np.array(((images_array==maxfiltered_image)*(images_array==maxfiltered_image_reversed[::-1,::-1])*(images_array>0)), dtype=np.int32)
    local_maxima[:,:int((aoi_size+1)/2)] = 0
    local_maxima[:,-int((aoi_size+1)/2):] = 0
    local_maxima[:int((aoi_size+1)/2),:] = 0
    local_maxima[-int((aoi_size+1)/2):,:] = 0

 

    (xcoords, ycoords) = np.where(local_maxima==1)
    local_maxima_coords = np.transpose(np.stack((ycoords, xcoords)))

    #subtract half an aoi size from local maxima locations to get top left corners of aois
    aoi_offset = int((aoi_size-1)/2)
    aoi_locations_array = local_maxima_coords - np.full(shape=local_maxima_coords.shape, fill_value=aoi_offset)

    aoi_locations = [str(x[0])+','+str(x[1]) for x in aoi_locations_array]
    return aoi_locations, images_array


def find_centroids(image, aoi_locations, aoi_size):

    centroids={}
    for aoi in aoi_locations:
        xmin=int(aoi.split(',')[0])
        ymin=int(aoi.split(',')[1])
        xmax, ymax = xmin+aoi_size, ymin+aoi_size
        centroid=center_of_mass(image[ymin:ymax, xmin:xmax])
        centroids[aoi]=centroid
    
    return centroids


def calculate_references(reference_image, aoi_locations, aoi_size):#

    references = find_centroids(reference_image, aoi_locations, aoi_size)

    return references


def find_slopes(centroids, references, focal_length, pixel_size, wavelength, magnification):
 
    
    differences = {}
    gradients = {}

    for location in references.keys():
        differences[location] = np.subtract(centroids[location], references[location])
        factor = 1/focal_length/magnification*pixel_size
        gradients[location] = np.multiply(factor, differences[location])

    return differences, gradients


def infer_aoi_size(aoi_locations):

        
    if(isinstance(aoi_locations, dict)):
        aoi_locations = np.array([[int(x) for x in aoi.split(',')] for aoi in aoi_locations])


    x_locations, y_locations = aoi_locations[:,0], aoi_locations[:,1]
    xmin, xmax, ymin, ymax = np.amin(x_locations), np.amax(x_locations), np.amin(y_locations), np.amax(y_locations)

    x_hist, x_bin_edges = np.histogram(x_locations, bins=int(xmax-xmin))
    y_hist, y_bin_edges = np.histogram(y_locations, bins=int(ymax-ymin))
    
    x_hist = x_hist-np.mean(x_hist)
    y_hist = y_hist-np.mean(y_hist)

    x_fft = np.fft.fft(x_hist)
    y_fft = np.fft.fft(y_hist)

    x_freq = np.fft.fftfreq(len(x_hist), d=1)
    y_freq = np.fft.fftfreq(len(y_hist), d=1)


    aoi_size_x = 1/np.abs(x_freq[np.argmax(x_fft)])
    aoi_size_y = 1/np.abs(y_freq[np.argmax(y_fft)])

    return aoi_size_x, aoi_size_y



def sort_aois(aoi_locations, aoi_size, buffer_size=3):

    
    if(isinstance(aoi_locations, dict)):
        aoi_locations = np.array([[int(x) for x in aoi.split(',')] for aoi in aoi_locations])

    if(isinstance(aoi_size, tuple)):
        aoi_size_x, aoi_size_y = aoi_size
    else:
        aoi_size_x = aoi_size
        aoi_size_y = aoi_size

    ymin, ymax = np.amin(aoi_locations[:,1]), np.amax(aoi_locations[:,1])
    
    sorted_list = []
    for y in np.arange(ymin, ymax+1, aoi_size_y):
        row = np.array(aoi_locations[np.logical_and(aoi_locations[:,1]+buffer_size>=y, aoi_locations[:,1]-buffer_size<=y)])
        sorted_row = row[np.argsort(row[:,0], axis=0)]
        sorted_list.append(sorted_row)
    
    sorted_aoi_locations = np.array([xy_tuple for row in sorted_list for xy_tuple in row])

    return sorted_aoi_locations


def get_current_aoi(aoi, aoi_locations):

    mask = [np.array_equal(x, aoi) for x in aoi_locations]
    
    return mask


def get_adjacent_aois(aoi, aoi_locations, aoi_size, buffer_size=3):

    
    x_mask_lower = np.logical_and(aoi_locations[:,0]+aoi_size+buffer_size>=aoi[0], aoi_locations[:,0]+aoi_size-buffer_size<=aoi[0])
    x_mask_upper = np.logical_and(aoi_locations[:,0]-aoi_size+buffer_size>=aoi[0], aoi_locations[:,0]-aoi_size-buffer_size<=aoi[0])
    x_mask_equal = np.logical_and(aoi_locations[:,0]+buffer_size>=aoi[0], aoi_locations[:,0]-buffer_size<=aoi[0])
    y_mask_lower = np.logical_and(aoi_locations[:,1]+aoi_size+buffer_size>=aoi[1], aoi_locations[:,1]+aoi_size-buffer_size<=aoi[1])
    y_mask_upper = np.logical_and(aoi_locations[:,1]-aoi_size+buffer_size>=aoi[1], aoi_locations[:,1]-aoi_size-buffer_size<=aoi[1])
    y_mask_equal = np.logical_and(aoi_locations[:,1]+buffer_size>=aoi[1], aoi_locations[:,1]-buffer_size<=aoi[1])
    
    x_mask = np.logical_and(np.logical_or(x_mask_lower, x_mask_upper), y_mask_equal)
    y_mask = np.logical_and(np.logical_or(y_mask_lower, y_mask_upper), x_mask_equal)
    mask = np.logical_or(x_mask, y_mask)
    
    return mask


def get_aoi_signature(aoi, adjacent_aoi, buffer_size=3):

    
    x_signature, y_signature = (0,0)
    
    if(adjacent_aoi[0]-buffer_size>aoi[0]+buffer_size):
        x_signature = 1
    elif(adjacent_aoi[0]+buffer_size<aoi[0]-buffer_size):
        x_signature = -1
        
    if(adjacent_aoi[1]-buffer_size>aoi[1]+buffer_size):
        y_signature = 1
    elif(adjacent_aoi[1]+buffer_size<aoi[1]-buffer_size):
        y_signature = -1
    
    signature = (x_signature, y_signature)
    
    return signature


def get_aoi_index(aoi, xmin, ymin, aoi_size, buffer_size=3):

    
    index = (int((aoi[0]-xmin+2*buffer_size)/aoi_size), int((aoi[1]-ymin+2*buffer_size)/aoi_size))
    
    return index


def to_string(aoi):

    
    string_representation = str(aoi[0])+','+str(aoi[1])
    return string_representation




def process_file(filepath, aoi_size, reference_image, focal_length, pixel_size, pitch_size , wavelength, magnification, no_div_conv=False, buffer_size=3):

    
    # NOISE FILTERING THRESHOLD. SUBJECT OF CHANGE FOR DIFFERENT DATASET
    noise_threshold = 0.1

    images_array = read_file(filepath)[0]
    images_array[images_array<noise_threshold*images_array.max()] = 0


    # FINDING SLOPES VIA FOURIER MODULATION
    if no_div_conv:

        # BELOW SIGNAL FILTERING VALUE EVERYTHING SETS TO ZERO!
        signal_filtering_x = 0.85
        signal_filtering_y = 0.75

        Nx = images_array.shape[1]
        Ny = images_array.shape[0]


        # FOURIER TRANSFORMATOIN - FIRST SILDLOBES WILL BE PROCESSED!
        I_f = ff(images_array)


        # FOURIER SPACE COORDINATE
        du_y = 2 * np.pi/(Ny*pixel_size)
        du_x = 2 * np.pi/(Nx*pixel_size)

        H_N_y = int(Ny/2)-1
        H_N_x = int(Nx/2)-1

        u_x = np.linspace((-H_N_x+1)*du_x,(H_N_x)*du_x,Nx)
        u_y = np.linspace((-H_N_y+1)*du_y,(H_N_y)*du_y,Ny)
        [ux,uy] = np.meshgrid(u_x,u_y)
        Q_mesh = [ux,uy]


        # FINDING PEAKS IN u_x and u_y DIRECTIONS
         
        I_f_c_x = np.abs(I_f[H_N_y,:])
        upper_limit = np.max(np.ndarray.flatten(I_f_c_x))
        I_f_c_x[I_f_c_x<signal_filtering_x*upper_limit] = 0
        peaks_x, _ = find_peaks(I_f_c_x, height=0)

        I_f_c_y = np.abs(I_f[:,H_N_x])
        upper_limit = np.max(np.ndarray.flatten(I_f_c_y))
        I_f_c_y[I_f_c_y<signal_filtering_y*upper_limit] = 0
        peaks_y, _ = find_peaks(I_f_c_y, height=0)


        # A QUASI SIDELOBE MAY EXIST AROUND PEAK - 
        # TO AVOID PEAKING IT DIFFERENCE BETWEEN CENTRAL AND 1st LOBE SETES TO LARGER THAN 10 PIXELS

        s_x_f = np.abs(peaks_x[1]-peaks_x[0])
        if s_x_f < 10:
            s_x_f = np.abs(peaks_x[2]-peaks_x[0])


        s_y_f = np.abs(peaks_y[1]-peaks_y[0])
        if s_y_f < 10:
            s_y_f = np.abs(peaks_y[2]-peaks_y[0])


        # LOBES ISOLATION

        lpf_x = pinholes(Q_mesh).CircPinhole(s_x_f*du_x,0,s_x_f*du_x)
        lpf_y = pinholes(Q_mesh).CircPinhole(s_y_f*du_y,s_y_f*du_y,0)

        F_I_x_is = I_f * lpf_x
        F_I_y_is = I_f * lpf_y


        # INVERS FOURIER TRANSFORM OF ISOLATED SIDELOBE

        Ph_F_I_x_is = iff(F_I_x_is)
        Ph_F_I_y_is = iff(F_I_y_is)


        # RECOVERING SLOPES:
        dph_x_wrap = np.angle(Ph_F_I_x_is) 
        dph_y_wrap = np.angle(Ph_F_I_y_is) 

        dph_x = unwrap_phase(dph_x_wrap)/np.pi * 4.0
        dph_y = unwrap_phase(dph_y_wrap)/np.pi * 4.0

        # REDUCE SLOPES DIMENSION TO (~) NUMBER OF WHOLES 

        dim = (37,37)
        dph_x = cv2.resize(dph_x, dim)
        dph_y = cv2.resize(dph_y, dim)

        dph_x  *= 1/focal_length/magnification*pixel_size
        dph_y  *= 1/focal_length/magnification*pixel_size

    
    else:
        dph_x = 0
        dph_y = 0


    # FINDING SLOPES CLASSICALLY

    aoi_locations, _ = get_aoi_locations(images_array, aoi_size)
    references = calculate_references(reference_image, aoi_locations, aoi_size)
  
    centroids = find_centroids(images_array, aoi_locations, aoi_size)
    _, differences = find_slopes(centroids, references, focal_length, pixel_size, wavelength, magnification)
    aoi_size_tuple = infer_aoi_size(references)
    sorted_aoi_locations = sort_aois(references,aoi_size_tuple,buffer_size=buffer_size)

    
    xmin, xmax = np.amin(sorted_aoi_locations[:,0]), np.amax(sorted_aoi_locations[:,0])
    ymin, ymax = np.amin(sorted_aoi_locations[:,1]), np.amax(sorted_aoi_locations[:,1])
    
    num_x, num_y = int((xmax-xmin+2*buffer_size)/aoi_size)+1, int((ymax-ymin+2*buffer_size)/aoi_size)+1
    
    differences_array = np.zeros((num_y, num_x,2))
    for aoi in sorted_aoi_locations:
        index = get_aoi_index(aoi, xmin, ymin, aoi_size)
        differences_array[index[1], index[0],0]= differences[to_string(aoi)][1]
        differences_array[index[1], index[0],1]= -differences[to_string(aoi)][0]

    return(differences_array[:,:],-dph_x,dph_y)



