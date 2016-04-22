#include "BackendTest.h"

TEST_F(BackendTest, MatchSuccess) {
    ASSERT_TRUE(bend_match_re("[0-9]+", "123"));
}

TEST_F(BackendTest, MatchFailure) {
    ASSERT_FALSE(bend_match_re("[0-9]+", "num"));
}
