// md5.cpp : Defines the entry point for the console application.
//

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include "pglib.h"

#include "apr_pools.h"
#include "apr_getopt.h"
#include "apr_strings.h"
#include "apr_md5.h"
#include "apr_file_io.h"

#define FILE_BUFFER_SIZE 262144
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"


void PrintCopyright(void) {
	CrtPrintf("\nMD5 Calculator\nCopyright (C) 2009 Alexander Egorov.  All rights reserved.\n\n");
}

void PrintUsage();
int CalculateMd5(apr_pool_t* pool, const char* file, apr_byte_t* digest);

int main(int argc, const char * const argv[])
{
	apr_pool_t* pool = NULL;
	apr_getopt_t* opt = NULL;
	char c = '\0';
	const char *optarg = NULL;
	const char *pFile = NULL;
	int isPrintLowCase = 0;
	int i = 0;
	apr_byte_t digest[APR_MD5_DIGESTSIZE];
	apr_status_t status = APR_SUCCESS;

	apr_app_initialize(&argc, &argv, NULL);
	apr_pool_create(&pool, NULL);
	apr_getopt_init(&opt, pool, argc, argv);

	while ((status = apr_getopt(opt, "?f:l", &c, &optarg)) == APR_SUCCESS) {
		switch(c){
			case '?':
				PrintUsage();
				goto cleanup;
			case 'f':
				pFile = apr_pstrdup(pool, optarg);
				break;
			case 'l':
				isPrintLowCase = 1;
				break;
		}
	}

	if (status != APR_EOF) {
		PrintUsage();
		goto cleanup;
	}

	if (pFile != NULL) {
		if (CalculateMd5(pool, pFile, digest)) {
			for (; i < APR_MD5_DIGESTSIZE; ++i) {
				CrtPrintf(isPrintLowCase ? HEX_LOWER : HEX_UPPER, digest[i]);
			}
			CrtPrintf("\n");
		}
	}

cleanup:
	apr_pool_destroy(pool);
	apr_terminate();
	return EXIT_SUCCESS;
}

void PrintUsage()
{
	PrintCopyright();
	CrtPrintf("usage: md5 [-f <PATH_TO_FILE>] [-l] [-?]\n");
}

int CalculateMd5(apr_pool_t* pool, const char* pFile, apr_byte_t* digest) {
	apr_file_t* file = NULL;
	apr_byte_t* pFileBuffer = NULL;
	apr_size_t readBytes = 0;
	apr_md5_ctx_t context = {0};
	apr_status_t status = APR_SUCCESS;
	int result = 1;
	
	status = apr_file_open(&file, pFile, APR_READ | APR_BUFFERED, APR_FPROT_WREAD, pool);
	if (status != APR_SUCCESS) {
		CrtPrintf("Failed to open file: %s\n", pFile);
		return 0;
	}				
	if(apr_md5_init(&context) != APR_SUCCESS) {
		CrtPrintf("Failed to initialize MD5 context\n");
		result = 0;
		goto cleanup;
	}

	pFileBuffer = (apr_byte_t*)apr_pcalloc(pool, FILE_BUFFER_SIZE);
	if (pFileBuffer == NULL) {
		CrtPrintf("Failed to allocate %i bytes from pool\n", FILE_BUFFER_SIZE);
		result = 0;
		goto cleanup;
	}

	do {
		status = apr_file_read_full(file, pFileBuffer, FILE_BUFFER_SIZE, &readBytes);
		if(status != APR_SUCCESS && status != APR_EOF) {
			CrtPrintf("Failed to read from file: %s\n", pFile);
			result = 0;
			goto cleanup;
		}
		if(apr_md5_update(&context, pFileBuffer, readBytes) != APR_SUCCESS ) {
			CrtPrintf("Failed to update MD5 context\n");
			result = 0;
			goto cleanup;
		}
	} while (status != APR_EOF);
	apr_md5_final(digest, &context);
cleanup:
	apr_file_close(file);
	return result;
}