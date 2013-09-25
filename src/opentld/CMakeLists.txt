
link_directories(${OpenCV_LIB_DIR})

include_directories(main
    ../3rdparty/asef
    ../libopentld/imacq
    ../libopentld/mftracker
    ../libopentld/tld
    ../3rdparty/cvblobs
    ../3rdparty/flandmark
    ${OpenCV_INCLUDE_DIRS})

if(NOT USE_SYSTEM_LIBS)
    include_directories(../3rdparty/libconfig)
endif(NOT USE_SYSTEM_LIBS)

#-------------------------------------------------------------------------------
# main
add_library(main
	#../3rdparty/asef/asef.h
    main/helper_functions.h
    main/helper_functions.cpp
    main/Config.cpp
    main/Gui.cpp
	main/Main_eye.cpp
    main/Settings.cpp
	main/Trajectory.cpp
    main/Config.h
    main/Gui.h
    main/Main.h
    main/Settings.h
	main/Trajectory.h)

target_link_libraries(main libopentld asef flandmark cvblobs config++ ${OpenCV_LIBS})

#-------------------------------------------------------------------------------
# opentld
add_executable(opentld
    OpenTLD.cpp)

target_link_libraries(opentld main asef libopentld flandmark cvblobs config++ ${OpenCV_LIBS})

install(TARGETS opentld DESTINATION bin)

#-------------------------------------------------------------------------------
# qopentld
if(BUILD_QOPENTLD)
    include_directories(qopentld
        ${CMAKE_CURRENT_BINARY_DIR})

    include(${QT_USE_FILE})
    add_definitions(${QT_DEFINITIONS})

    set(QOPENTLD_SOURCES QOpenTLD.cpp qopentld/ConfigDialog.cpp)
    set(QOPENTLD_HEADERS qopentld/ConfigDialog.h)
    set(QOPENTLD_FORMS qopentld/ConfigDialog.ui)
    set(QOPENTLD_RCS ../../res/rcs.qrc)

    QT4_ADD_RESOURCES(QOPENTLD_RCS_CPP ${QOPENTLD_RCS})
    QT4_WRAP_CPP(QOPENTLD_HEADERS_MOC ${QOPENTLD_HEADERS})
    QT4_WRAP_UI(QOPENTLD_FORMS_HEADERS ${QOPENTLD_FORMS})
    set(QOPENTLD_SOURCES ${QOPENTLD_SOURCES} ${QOPENTLD_RCS_CPP} ${QOPENTLD_HEADERS_MOC} ${QOPENTLD_FORMS_HEADERS})

    add_executable(qopentld ${QOPENTLD_SOURCES})
    target_link_libraries(qopentld main libopentld asef flandmark cvblobs config++ ${OpenCV_LIBS} ${QT_LIBRARIES})

    install(TARGETS qopentld DESTINATION bin)
endif(BUILD_QOPENTLD)

