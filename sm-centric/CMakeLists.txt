set(LLVM_LINK_COMPONENTS support)

add_clang_executable(sm-centric
  SM-centric_transformation.cpp
    )
    target_link_libraries(sm-centric
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
