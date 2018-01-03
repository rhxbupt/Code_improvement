//------------------------------------------------------------------------------
// The transformation code for CUDA to SM-Centric
//
//
// Haoxu Ren(hren3@ncsu.edu)
//
//------------------------------------------------------------------------------
#include <string>
#include <iostream>
#include <fstream>
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Lex/PPCallbacks.h"
using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;
static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

// This VarDeclhandle designed for handling matched VariableDeclaration
class VarDeclHandler : public MatchFinder::MatchCallback {
public:
    VarDeclHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

    virtual void run(const MatchFinder::MatchResult &Result) {
        //get the node "Vardecl"
        if(const MemberExpr *Var = Result.Nodes.getNodeAs<MemberExpr>("Vardecl")){
            //get the right side name of blockIdx.x
            std::string name = Var->getMemberDecl()->getNameAsString();
            //if the right side's name is x
            if(name == "__fetch_builtin_x")
                //replace the "blockIdx.x" with "(int)fmodf((float)__SMC_chunkID, (float)__SMC_orgGridDim.x)"
                Rewrite.ReplaceText(Var->getLocEnd().getLocWithOffset(-9),11,"(int)fmodf((float)__SMC_chunkID, (float)__SMC_orgGridDim.x);\n");
            // if the right side's name is y
            if(name == "__fetch_builtin_y")
                //replace the "blockIdx.y" with "(int)(__SMC_chunkID/__SMC_orgGridDim.x)"
                Rewrite.ReplaceText(Var->getLocEnd().getLocWithOffset(-9),11,"(int)(__SMC_chunkID/__SMC_orgGridDim.x);\n");
        }

        //get the node "grid"
        if(const VarDecl *Var = Result.Nodes.getNodeAs<VarDecl>("grid")){
            // get the sourcelocation of the call of function "grid(...)"
            SourceLocation sr = Var->getLocStart().getLocWithOffset(4);
            // replace the 'grid(...)' with "dim3 __SMC_orgGridDim(...)"
            Rewrite.ReplaceText(sr,Var->getNameAsString().length()," __SMC_orgGridDim");
        }
    }

private:
    Rewriter &Rewrite;
};
// This FunctionDeclHandle designed for handling matched FunctionDeclaration
class FunctionDeclHandler : public MatchFinder::MatchCallback {
public:
    FunctionDeclHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

    virtual void run(const MatchFinder::MatchResult &Result) {
        //get the node 'kernelfunc'
        if(const FunctionDecl *FuncDecl = Result.Nodes.getNodeAs<FunctionDecl>("kernalfunc")){
            // if this function is cuda kernel function
            if(FuncDecl->hasAttr<CUDAGlobalAttr>()){
                // get the function body of this cuda kernel function
                const Stmt *FuncBody = FuncDecl->getBody();
                // Insert the "_SMC_Begin" to the top of the cuda kernel function
                Rewrite.InsertTextAfterToken(FuncBody->getLocStart(), "\n__SMC_Begin;\n");
                // Insert the "_SMC_End" to the bottom of the cuda kernel function
                Rewrite.InsertText(FuncBody->getLocEnd(), "\n__SMC_End; \n", true, true);
                // Initialize a size_t for storing the number of current parameters
                size_t a;
                a=FuncDecl->param_size();
                // get the current parameters of this kernel function
                const ParmVarDecl *para = FuncDecl->getParamDecl(a-1);
                //Add the "dim3 __SMC_orgGridDim, int __SMC_workersNeeded, int *__SMC_workerCount, int * __SMC_newChunkSeq, int * __SMC_seqEnds"
                // to the end of the argument list of the definition of the kernel function
                Rewrite.InsertTextAfterToken(para->getLocEnd(),", dim3 __SMC_orgGridDim, int __SMC_workersNeeded, int *__SMC_workerCount, int * __SMC_newChunkSeq, int * __SMC_seqEnds");
            }
        }
    }

private:
    Rewriter &Rewrite;
};
// This CudaCallPrinter designed for handling matched Cuda kernel function call.
class CudaCallPrinter : public MatchFinder::MatchCallback {
public :
    CudaCallPrinter(Rewriter &Rewrite) : Rewrite(Rewrite) {}

    virtual void run(const MatchFinder::MatchResult &Result) {
        //get the node 'cudaCall'
        if (const Stmt *S = Result.Nodes.getNodeAs< clang::Stmt>("cudaCall")){
            // Insert the "_SMC_init()" right before the call of the GPU kernel function
            Rewrite.InsertText(S->getLocStart(), "\n __SMC_init();\n",true,true);
            // if this function call is the call of CUDA kernel function
            if( const CUDAKernelCallExpr *cudafunc =  dyn_cast< CUDAKernelCallExpr>(S)){
                // get the location of this call's arguments
                const Stmt *arg = cudafunc->getArg(cudafunc->getNumArgs()-1);
                // append the following arguments to the end of the GPU kernel function call
                Rewrite.InsertText(arg->getLocStart().getLocWithOffset(1), ", __SMC_orgGridDim, __SMC_workersNeeded, __SMC_workerCount,__SMC_newChunkSeq, __SMC_seqEnds",true,true);
            }
        }

    }
private:
    Rewriter &Rewrite;
};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser. It registers a couple of matchers and runs them on
// the AST.
class MyASTConsumer : public ASTConsumer {
public:
    MyASTConsumer(Rewriter &R) : cudaPrinter(R), HandleForFunction(R),HandleForDecl(R) {
        // The matcher of kernel function
        Matcher.addMatcher(
                functionDecl().bind("kernalfunc"),&HandleForFunction);
        // The matcher of blockIdx.x
        Matcher.addMatcher(
                memberExpr(hasObjectExpression(hasType(asString("const struct __cuda_builtin_blockIdx_t")))).bind("Vardecl"),&HandleForDecl);
        // The matcher of grid() declaration
        Matcher.addMatcher(
                varDecl(hasType(asString("dim3")),hasName("grid")).bind("grid"),&HandleForDecl);
        // The matcher of the call of Cuda kernel function
        Matcher.addMatcher(
                cudaKernelCallExpr().bind("cudaCall"), &cudaPrinter);

    }

    void HandleTranslationUnit(ASTContext &Context) override {
        // Run the matchers when we have the whole TU parsed.
        Matcher.matchAST(Context);
    }

private:
    CudaCallPrinter cudaPrinter;
    FunctionDeclHandler HandleForFunction;
    VarDeclHandler HandleForDecl;
    MatchFinder Matcher;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
    MyFrontendAction() {}
    void EndSourceFileAction() override {
        const RewriteBuffer *Rewritebuffer = TheRewriter.getRewriteBufferFor(TheRewriter.getSourceMgr().getMainFileID());
        // Initilize the output file.
        std::ofstream Outfile;
        std::string filename = std::string(getCurrentFile());
        filename.insert(filename.length()-3,"_smc");
        Outfile.open(filename);
        // add the #include "smc.h"  to he beginning of  the CUDA file
        Outfile<<"#include \"smc.h\""<<"\n";
        // Rewrite the cuda file to SM-Centric format.
        Outfile<<std::string(Rewritebuffer->begin(),Rewritebuffer->end());
        Outfile.close();
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef file) override {
        TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return llvm::make_unique<MyASTConsumer>(TheRewriter);
    }

private:
    Rewriter TheRewriter;
};
// The main function
int main(int argc, const char **argv) {
    CommonOptionsParser op(argc, argv, MatcherSampleCategory);
    ClangTool Tool(op.getCompilations(), op.getSourcePathList());

    return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}

