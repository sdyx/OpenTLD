/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

/*
 * main.h
 *
 *  Created on: Nov 18, 2011
 *      Author: Georg Nebehay
 *	Extended on: Sept. 12, 2013
 *		Author: Felix Baumann
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "TLD.h"
#include "ImAcq.h"
#include "Gui.h"
#include "asef_type.h"
#include "asef_new.h"

#include <cstdio>

enum Retval
{
    PROGRAM_EXIT = 0,
    SUCCESS = 1
};

class Main
{
public:
    tld::TLD *tld;
    //FB
    tld::TLD *eyeTld;
    AsefEyeLocator asef;
    double average;
    //FB end
    ImAcq *imAcq;
    tld::Gui *gui;
    bool showOutput;
	bool showTrajectory;
	int trajectoryLength;
    const char *printResults;
    const char *saveDir;
    double threshold;
    bool showForeground;
    bool showNotConfident;
    bool selectManually;
    int *initialBB;
    bool reinit;
    bool exportModelAfterRun;
    bool loadModel;
    const char *modelPath;
    const char *modelExportFile;
    int seed;

    Main()
    {
        tld = new tld::TLD();
        //FB
        eyeTld = new tld::TLD();
        
        
	const char *asef_locator_path = "EyeLocatorASEF128x128.fel";
        ASEF_initialze( &asef,	asef_locator_path );
	
	
	
        average = 0.0;
        showOutput = 1;
        printResults = NULL;
        saveDir = ".";
        threshold = 0.5;
        showForeground = 0;

		showTrajectory = false;
		trajectoryLength = 0;

        selectManually = 0;

        initialBB = NULL;
        showNotConfident = true;

        reinit = 0;

        loadModel = false;

        exportModelAfterRun = false;
        modelExportFile = "model";
        seed = 0;
        
		currentEyeMean = 0.0;
		eyeMeanThreshold = 0;
		averageEyeMean = 0.0;
    }

    ~Main()
    {
        delete tld;
        delete eyeTld;
        imAcqFree(imAcq);
    }

    void doWork();
    std::vector<cv::Point>  doFlandmark( IplImage image,
			cv::Rect& detectionArea
	);
    void doHaar( cv::Mat image );
    double calculateVariance( cv::Mat& image );
    cv::Rect doDetectEye( cv::Mat image );
	cv::Point findPupil( cv::Mat image,
			cv::Rect faceBoundingBox,
			cv::Rect eyeBoundingBox
	);
	
	/*
	void ASEF_locate_eyes( AsefEyeLocator *asef, CvRect area );
	int ASEF_detect_face( AsefEyeLocator *asef, CvRect area );
	int ASEF_read_line( FILE* fp, char* buf, int size );
	void ASEF_destroy( AsefEyeLocator *asef );
	int ASEF_load_filters( const char* file_name,
			int *p_n_rows,
			int *p_n_cols, 
			CvRect *left_eye_region,
			CvRect *right_eye_region, 
			CvMat **p_left_filter,
			CvMat **p_right_filter
	);
	int ASEF_initialze( AsefEyeLocator *asef,
			const char *asef_file_name);
						*/
private:
	static const double lowerBlinkBound = 0.95;
	static const double upperBlinkBound = 1.05;
	double currentEyeMean;
	double averageEyeMean;
	int eyeMeanThreshold;
};

#endif /* MAIN_H_ */
