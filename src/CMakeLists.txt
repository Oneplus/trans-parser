add_library (tp_layer layer.cc layer.h)
add_library (tp_noisify noisify.cc noisify.h)
add_library (tp_evaluate evaluate.cc evaluate.h)

add_library (tp_system
    arcstd.cc arcstd.h
    arceager.cc arceager.h
    archybrid.cc archybrid.h
    swap.cc swap.h
    system.cc system.h
    system_builder.cc system_builder.h)

add_library (tp_dataset
    alphabet.cc alphabet.h
    corpus.cc corpus.h tree.h)

add_library (tp_utils
    logging.cc logging.h
    sys_utils.cc sys_utils.h
    math_utils.cc math_utils.h
    trainer_utils.cc trainer_utils.h)

 add_library (tp_parser
    parser.cc parser.h
    parser_dyer15.cc parser_dyer15.h
    parser_ballesteros15.cc parser_ballesteros15.h
    parser_kiperwasser16.cc parser_kiperwasser16.h
    parser_builder.cc parser_builder.h)

add_library (tp_train
    train_supervised.cc train_supervised.h 
    train_supervised_ensemble.cc train_supervised_ensemble.h)
    
add_executable(trans_parser main.cc)
add_executable(trans_parser_ensemble ensemble.cc)

if (MSVC)
    target_link_libraries(tp_utils dynet ${LIBS})
    target_link_libraries(tp_layer dynet ${LIBS})
    target_link_libraries(tp_parser tp_system tp_layer dynet ${LIBS})
    target_link_libraries(tp_train tp_dataset tp_utils tp_parser tp_noisify dynet ${LIBS})
    target_link_libraries(tp_evaluate tp_parser dynet ${LIBS})
    target_link_libraries(trans_parser dynet tp_system tp_dataset tp_utils tp_layer tp_noisify tp_parser tp_train tp_evaluate ${LIBS})
    target_link_libraries(trans_parser_ensemble dynet tp_dataset tp_system tp_utils tp_layer tp_noisify tp_parser tp_train tp_evaluate ${LIBS})
else()
    target_link_libraries(tp_utils dynet ${LIBS} z)
    target_link_libraries(tp_layer dynet ${LIBS} z)
    target_link_libraries(tp_parser tp_system tp_layer dynet ${LIBS} z)
    target_link_libraries(tp_train tp_dataset tp_utils tp_parser tp_noisify dynet ${LIBS} z)
    target_link_libraries(tp_evaluate tp_parser dynet ${LIBS} z)
    target_link_libraries(trans_parser dynet tp_dataset tp_system tp_utils tp_layer tp_noisify tp_parser tp_train tp_evaluate ${LIBS} z)
    target_link_libraries(trans_parser_ensemble dynet tp_dataset tp_system tp_utils tp_layer tp_noisify tp_parser tp_train tp_evaluate ${LIBS} z)
endif()
