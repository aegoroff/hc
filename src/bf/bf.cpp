// bf.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

char *bruteforce(int passmax, int passmin);

int count = 0;
char *passwd;

#define MAXCHAR 50
#define MINCHAR 48

/* this code is copyrighted by Hi_tech_assassin, see pscode.com */

char *bruteforce(int passmax, int passmin)
{
	char *pass = (char *)malloc(passmin + 1);
	int position, x, found;

	/* since we can only do one increment per 
	   iteration we need a way of controling this */

	memset(pass, MINCHAR, passmin);
	pass[passmin] = '\0';

	for (x = passmin; x <= passmax; ++x) {
		if (x > passmin) {
			realloc(pass, x + 1);
			memset(pass, MINCHAR, x);
			pass[x] = '\0';
		}

		while (pass[0] <= MAXCHAR) {
			found = 0;
			printf("%s\n", pass);
            if (strcmp(pass, passwd) == 0) {
                return _strdup(pass);
            }

			/* you may replace this function with the encryption function
			   and return in same manner if the encryption string matches
			   the one we are cracking */

			for (position = x - 1; position > 0; --position) {
				if (pass[position] == MAXCHAR) {
                    memset(pass + position, MINCHAR, x - position);
                    pass[position - 1]++;
					found = 1;
					break;
				}
			}

			if (!found){
				pass[x - 1]++;
			}

			count++;
		}
	}
	//free(pass);
	return NULL;
}


int main(int argc, char *argv[])
{
	char *pass;
	clock_t start, end;
	double elapsed;

	if (argc != 2) {
		printf("\nProof of concept brute force\nby Hi Tech Assassin\n");
		printf("\nUsage: %s <pass to crack>\n\n", argv[0]);
		return 0;
	}

	start = clock();
	passwd = argv[1];
	printf("\nAttempting to brute force \"%s\"\n", passwd);
	printf("Should take approximatelly around %.0f trys\n\n", pow((double)(MAXCHAR - MINCHAR), (int)strlen(passwd)));

	if (pass = bruteforce(strlen(passwd), 1)) {
		printf("The correct password is \"%s\"", pass);
	} else {
		printf("Hard luck\n");
	}

	printf("\nNo of trys was %d\n", count);
	end = clock();
	elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time elapsed = %f seconds\n", elapsed);

	if (elapsed >= 1) {
		printf("Trys per second was %f\n\n", count / elapsed);
	} else {
		printf("\n");
	}

	return 0;
}
