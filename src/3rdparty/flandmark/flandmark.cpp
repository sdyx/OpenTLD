#include "flandmark_detector.h"

#include "flandmark.h"

void useFlandmark()
{
// load flandmark model structure and initialize
	    FLANDMARK_Model * model = flandmark_init( "flandmark_model.dat" );

	    // bbox with detected face 
	    // (format: top_left_col top_left_row bottom_right_col bottom_right_row)
	    int bbox[] = { 72, 72, 183, 183 };

	    // detect facial landmarks (output are x, y coordinates
	    // of detected landmarks)
	    float * landmarks = ( float* )malloc(
	    		2 * model->data.options.M * sizeof( float ) );
	    flandmark_detect( img_grayscale,
	    	bbox,
	    	model,
	    	landmarks
		);
}

