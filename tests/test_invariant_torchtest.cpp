#include <gtest/gtest.h>
#include <cstdio>
#include <cstring>
#include <vector>

// External declaration of the vulnerable function from trainer/torchtest.cpp
extern void formatEpochFilename(char* buffer, int bufferSize, int epoch);

class BufferOverflowSecurityTest : public ::testing::TestWithParam<int> {};

TEST_P(BufferOverflowSecurityTest, BufferReadNeverExceedsDeclaredLength) {
    // Invariant: Buffer reads never exceed the declared length
    // Even with adversarial epoch values, no out-of-bounds access occurs
    
    int epoch = GetParam();
    const int BUFFER_SIZE = 256;
    char buffer[BUFFER_SIZE];
    
    // Fill buffer with sentinel pattern to detect overflow
    memset(buffer, 0xAA, BUFFER_SIZE);
    
    // Call the vulnerable function with adversarial input
    formatEpochFilename(buffer, BUFFER_SIZE, epoch);
    
    // Verify buffer bounds: check that sentinel bytes beyond expected output remain unchanged
    // Expected max output is roughly 50 bytes for typical filename format
    const int EXPECTED_MAX_OUTPUT = 100;
    ASSERT_LT(EXPECTED_MAX_OUTPUT, BUFFER_SIZE);
    
    // Check that bytes beyond expected output are not corrupted
    for (int i = EXPECTED_MAX_OUTPUT; i < BUFFER_SIZE; i++) {
        EXPECT_EQ(buffer[i], 0xAA) 
            << "Buffer overflow detected at offset " << i 
            << " with epoch=" << epoch;
    }
    
    // Verify null termination within bounds
    bool null_found = false;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        if (buffer[i] == '\0') {
            null_found = true;
            EXPECT_LT(i, EXPECTED_MAX_OUTPUT) 
                << "String length exceeds expected bounds with epoch=" << epoch;
            break;
        }
    }
    EXPECT_TRUE(null_found) << "Buffer not null-terminated with epoch=" << epoch;
}

INSTANTIATE_TEST_SUITE_P(
    AdversarialEpochValues,
    BufferOverflowSecurityTest,
    ::testing::Values(
        1,                    // Valid input: normal epoch
        999999,               // Boundary: large epoch number (6 digits)
        2147483647,           // Exploit: max int32 (10 digits)
        -1,                   // Boundary: negative epoch
        1000000000            // Boundary: very large epoch
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}