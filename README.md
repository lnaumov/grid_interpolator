# grid_interpolator
This repository contains a script interpolating masked model data (1D, 2D, 3D, or 4D) from one regular model grid to another.
# Usage:
1. **Make sure you have the following python libraries installed:**
* numpy;
* scipy;
* netCDF4;
* xarray.

2. **Choose whether you like to work with Jupyter notebbok file or simple .py file. Depending on your preference, use either .ipynb or .py file.**
3. **Initiate object *gridded_fields_interpolation* here is its the description:**

        This object performs interpolation procedure from one model grid to another e.g. reanalysis grid. 
        The majority of grid points should be interpolated by 3D/2D/1D linear interpolation, for some grid points nearest neighbor interpolation is performed.

        To get started only one parameter is needed: 
        Path to the grid file - path containinig the netCDF file where some necessary grid information is stored. 

        For MOM grid it is grid_spec.nc file.

        For other grids make sure the file contains the following information:
        1) Time, longitude, latitue, depth - arrays representing time, longitude, latitude and depth;
        2) One arbitrary variable written on that grid (e.g. temperature or salinity).

        Important info for not MOM grids: 
    
        1. Even if the data is not 4D (time/depth is/are missing) you should have these dimensions and variables in the netCDF file single valued.
        Example: 2D SST data should have depth 0 and an arbitrary timestep, say 01-01-1970.
        2. Make sure the order of the dimensions is: (time, depth, latitdute, longitude).
        Example: variable temperature has 5 timesteps, 10 depth levels, 30 latitudes and 50 longitudes. Correct shape for that variable will be: (5,10,30,50) not (10,5,50,30) or another combination.
    
        As an output two netCDF files will be written:
    
        1. File containing interpolation results;
        2. File called interpolation_log.nc, which will have some technical information about how interpolation has been performed. 
    
        Example of usage:
    
        grid_interpolation = gridded_fields_interpolation(path_to_grid_file)
        grid_interpolation.interpolate_fields_to_model_grid(args)
    
4. **Prepare a dictionary with the output parameters you would like to have. It can be done with *generate_vars_dictionary* function. You can find its description below:**

 
        This function will generate a variables dictionary formatted in a way that will be recognizible by the interpolation function
    
        Arguments:
    
        input_names - any iterable containing variable names you want to be interpolated;
    
        output_names - any iterable containing names you want to be in your interpolated file;
    
        Warning: input_names and output_names need to be consistent e.g. input_names are ['temperature', 'pressure'], output_names are ['t2m', 'p']
    
        path2save - any iterable containing the path to file you want to save your results to (with file name and extension)
    
        grid_type - any iterable containing the type of grid: 't' for t-grid or 'u' for u-grid
    
5. **Execute *interpolate_fields_to_model_grid* method. Its documentaion is provided below:**

        Interpolate data to previously defined grid.
        
        Required arguments are:
        
        aux_variables - any iterable with names of auxilary variables (time, depth etc.) in the input file;
        Note: The order is important. It should be: 'longitude', 'latitude', 'depth', 'time'
        Example: ['lon', 'lat', 'z', 't']
        
        variables - dictionary with some important information regarding the variables subjected to interpolation (created using generate_vars_dictionary function);
        
        path2init_fields - path to the file containing the fields that needed to be interpolated.
        
        Optional arguments are:
        
        crop - any iterable containing coordinates of area corners (only if cropping of model grid (the grid to interpolate on) is needed);
        The format of the any iterable is: [min_lat, max_lat, min_lon, max_lon]
        Default: None
        
        parallel - whether parallelization is needed. Number of processes will be calculated authomatically;
        Default: True
        
        to_mom - whether it is planned to interpolate fields to the MOM grid;
        Default: True
        
        Next two parameters only applicable if to_mom is set to False:
        
        dim_names - any iterable containing names of the dimensions in a not-MOM grid file;
        Note: The order is important. It should be: 'longitude', 'latitude', 'depth', 'time';
        Example: ['x_grid', 'y_grid', 'z_grid', 't_grid']
        Default: False
        
        var_names - any iterable containing names of the auxilary variables in a not-MOM grid file (could be the same names as dimensions).
        Last element of the any iterable needs to be the name of an arbitrary variable stored in the grid netCDF file e.g. temperature;
        Note: The order is important. It should be: 'longitude', 'latitude', 'depth', 'time', 'arbitrary_variable';
        Example: ['lons', 'lats', 'depths', 'times', 'temp_ocean']
        Default: False
   
   I hope it will help you with your interpolation problem!
