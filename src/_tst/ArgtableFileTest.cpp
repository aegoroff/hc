#include "ArgtableFileTest.h"
#include "argtable3.h"

void ArgtableFileTest::SetUp() {
    a = arg_file1(nullptr, nullptr, "<file>", "filename to test");
    auto end = arg_end(20);
    n = 1;
    argtable = static_cast<void**>(malloc(n * sizeof(arg_dbl *) + sizeof(struct arg_end *)));
    argtable[0] = a;
    argtable[1] = end;
}

void ArgtableFileTest::TearDown() {
    arg_freetable(argtable, n + 1);
}
