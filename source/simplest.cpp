#include <stdio.h>
#include <slang.h>
#include <slang-com-ptr.h>

int main(int argc, char* argv[])
{
    printf("Starting creating device\n");
    SlangSession* globalSession = spCreateSession();
    printf("C API: slang session: %p\n", globalSession);


    Slang::ComPtr<slang::IGlobalSession> slangGlobalSession;
    slang::createGlobalSession(slangGlobalSession.writeRef());
    printf("C++ API: slang session: %p\n", slangGlobalSession.get());
    return 0;
}
