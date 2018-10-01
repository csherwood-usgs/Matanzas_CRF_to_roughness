# Matanzas_CRF_to_Roughness
Convert classes defined using Dan's CRF segmentation routine to Manning's n roughness values and interpolate onto the Delft XBeach grid.

This notebook demos an approach for converting images segmented by pixels to Manning's n for use as input to models.

The imagery was downloaded from NAIP and exported as a geotiff from Global Mapper

It was classified using a version of Dan Buscomb's earlier CRF program ```int_seg_crf_matanzas.py``` routine, modified to look for the following classes:

```classes = {'sand':'d', 'wetland veg':'g', 'water':'y', 'dune grass':'v', 'woody vegetation':'n','anthropogenic':'q'}```

The results were saved as a .mat file.

Steps are:
 * Load the image and determine projection and coordinates
 * Load the .mat file produced by ```int_seg_crf```
 * Make a look-up table to convert classes to Manning's n
 * Use the lookup table to create a grid of roughness values
 * [Smooth or edit the roughness array?]
 * Load the model grid coordinates
 * Transform model grid coordinates projection of roughness array
 * Interpolate roughness values onto model grid
 * Save model grid
 
