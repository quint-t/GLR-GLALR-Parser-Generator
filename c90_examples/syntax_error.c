// Single line comment
/*
Multi line comment block
*/
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
static enum Enum { VAR_1 = 0, VAR_2, VAR_3 = 0, VAR_4, VAR_5 = 0, VAR_6 } some_enum1, some_enum2;
const volatile int global_var = 1 + 1 << sizeof(char), another_var = 2 * global_var;
static struct Animal {
    int animal_int : 32;
    const void * volatile animal_reference;
} some_animal1, some_animal2;
int main(int _, ...) {
    int * const r = (int*) malloc(sizeof(int) * 5), k = (2 >> 1) + (1 << 1) - 1 * 1.0e-5/1e+20;
    wprintf(L"%d", ((0 <= 0 < 2 > 0 >= 0) == 1 != 0));
    ((((k += 1) -= 1) *= 1) /= 1 % 2) %= 2;
    ++k; k++; --k; k--;
    ((k &= 1 || 1 | 1 & 5 && 5) |= ~k ^ k) ^= !k;
    do {
        for (k = 0; k < 5; ++k) {
            wprintf(L"\nHello, world\n");
        }
        while (++k < 10) continue;
    }
    while (0);
    fputwc(L'\n', stdout);
    free(r);
    return 0
}