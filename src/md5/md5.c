/*
 * Copyright 2009 Alexander Egorov
 */

#include "targetver.h"

#include <stdio.h>
#include "pglib.h"

#include "apr_pools.h"
#include "apr_getopt.h"
#include "apr_strings.h"
#include "apr_md5.h"
#include "apr_file_io.h"

#define FILE_BUFFER_SIZE 262144
#define FILE_BIG_BUFFER_SIZE 8 * 1024 * 1024 // 8 megabytes
#define ERROR_BUFFER_SIZE 2048
#define BYTE_CHARS_SIZE 2 // byte representation string length
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"
#define HLP_ARG "  -%c [ --%s ] arg\t\t%s\n"
#define HLP_NO_ARG "  -%c [ --%s ] \t\t%s\n"

static struct apr_getopt_option_t options[] = {
	{ "file", 'f', TRUE, "input full file path to calculate MD5 sum for" },
	{ "string", 's', TRUE, "string to calculate MD5 sum for" },
	{ "md5", 'm', TRUE, "MD5 hash to validate file" },
	{ "lower", 'l', FALSE, "whether to output sum using low case" },
	{ "help", '?', FALSE, "show help message" }
};

void PrintCopyright(void) {
	CrtPrintf("\nMD5 Calculator\nCopyright (C) 2009-2010 Alexander Egorov. All rights reserved.\n\n");
}

void PrintUsage();
int CalculateFileMd5(apr_pool_t* pool, const char* file, apr_byte_t* digest);
int CalculateStringMd5(const char* string, apr_byte_t* digest);
void PrintMd5(apr_byte_t* digest, int isPrintLowCase);
void CheckMd5(apr_byte_t* digest, const char* pCheckSum);
void PrintError(apr_status_t status);

int main(int argc, const char * const argv[]) {
	apr_pool_t* pool = NULL;
	apr_getopt_t* opt = NULL;
	int c = 0;
	const char *optarg = NULL;
	const char *pFile = NULL;
	const char *pCheckSum = NULL;
	const char *pString = NULL;
	int isPrintLowCase = FALSE;
	apr_byte_t digest[APR_MD5_DIGESTSIZE];
	apr_status_t status = APR_SUCCESS;

	if (argc < 2) {
		PrintUsage();
		return EXIT_SUCCESS;
	}

	status = apr_app_initialize(&argc, &argv, NULL);
	if (status != APR_SUCCESS) {
		CrtPrintf("Could't initialize APR\n");
		PrintError(status);
		return EXIT_FAILURE;
	}
	atexit(apr_terminate);
	apr_pool_create(&pool, NULL);
	apr_getopt_init(&opt, pool, argc, argv);

	while ((status = apr_getopt_long(opt, options, &c, &optarg)) == APR_SUCCESS) {
		switch(c){
			case '?':
				PrintUsage();
				goto cleanup;
			case 'f':
				pFile = apr_pstrdup(pool, optarg);
				break;
			case 'm':
				pCheckSum = apr_pstrdup(pool, optarg);
				break;
			case 's':
				pString = apr_pstrdup(pool, optarg);
				break;
			case 'l':
				isPrintLowCase = TRUE;
				break;
		}
	}

	if (status != APR_EOF) {
		PrintUsage();
		goto cleanup;
	}

	if (pFile != NULL && pCheckSum == NULL && CalculateFileMd5(pool, pFile, digest)) {
		PrintMd5(digest, isPrintLowCase);
	}
	if (pString != NULL && CalculateStringMd5(pString, digest)) {
		PrintMd5(digest, isPrintLowCase);
	}
	if (pCheckSum != NULL && pFile != NULL && CalculateFileMd5(pool, pFile, digest)) {
		CheckMd5(digest, pCheckSum);
	}

cleanup:
	apr_pool_destroy(pool);
	return EXIT_SUCCESS;
}

void PrintError(apr_status_t status) {
	char errbuf[ERROR_BUFFER_SIZE];
	apr_strerror(status, errbuf, ERROR_BUFFER_SIZE);
	CrtPrintf("%s\n", errbuf);
}

void PrintUsage() {
	int i = 0;
	PrintCopyright();
	CrtPrintf("usage: md5 [OPTION] ...\n\nOptions:\n\n");
	for(; i < sizeof(options) / sizeof(apr_getopt_option_t); ++i) {
		CrtPrintf(
			options[i].has_arg ? HLP_ARG : HLP_NO_ARG, 
			(char)options[i].optch,
			options[i].name,
			options[i].description
			);
	}
}

void PrintMd5(apr_byte_t* digest, int isPrintLowCase) {
	int i = 0;
	for (; i < APR_MD5_DIGESTSIZE; ++i) {
		CrtPrintf(isPrintLowCase ? HEX_LOWER : HEX_UPPER, digest[i]);
	}
	CrtPrintf("\n");
}

void CheckMd5(apr_byte_t* digest, const char* pCheckSum) {
	char digestString[APR_MD5_DIGESTSIZE * BYTE_CHARS_SIZE + 1];
	int i = 0;

	digestString[APR_MD5_DIGESTSIZE * BYTE_CHARS_SIZE] = 0; // trailing zero
	for (; i < APR_MD5_DIGESTSIZE; ++i) {
		apr_snprintf(
			digestString + i * BYTE_CHARS_SIZE, 
			BYTE_CHARS_SIZE + 1, // trailing zero
			HEX_UPPER, 
			digest[i]);
	}
	if (apr_strnatcasecmp(pCheckSum, digestString) == 0) {
		CrtPrintf("\nFile is valid!\n");
	
	} else {
		CrtPrintf("\nFile is invalid!\n");
	}
}

int CalculateFileMd5(apr_pool_t* pool, const char* pFile, apr_byte_t* digest) {
	apr_file_t* file = NULL;
	apr_finfo_t info;
	apr_byte_t* pFileBuffer = NULL;
	apr_size_t readBytes = 0;
	apr_md5_ctx_t context = {0};
	apr_status_t status = APR_SUCCESS;
	apr_status_t md5CalcStatus = APR_SUCCESS;
	int result = TRUE;
	size_t bufferSize = FILE_BUFFER_SIZE;
	
	status = apr_file_open(&file, pFile, APR_READ | APR_BUFFERED, APR_FPROT_WREAD, pool);
	if (status != APR_SUCCESS) {
		PrintError(status);
		return FALSE;
	}				
	status = apr_md5_init(&context);
	if(status != APR_SUCCESS) {
		PrintError(status);
		result = FALSE;
		goto cleanup;
	}

	status = apr_file_info_get(&info, APR_FINFO_SIZE, file);

	if(status != APR_SUCCESS) {
		PrintError(status);
		result = FALSE;
		goto cleanup;
	}

	if (info.size > FILE_BIG_BUFFER_SIZE) {
		bufferSize = FILE_BIG_BUFFER_SIZE;
	}

	pFileBuffer = (apr_byte_t*)malloc(bufferSize);
	if (pFileBuffer == NULL) {
		CrtPrintf("Failed to allocate %i bytes\n", bufferSize);
		result = FALSE;
		goto cleanup;
	}

	do {
		status = apr_file_read_full(file, pFileBuffer, bufferSize, &readBytes);
		if(status != APR_SUCCESS && status != APR_EOF) {
			CrtPrintf("Failed to read from file: %s\n", pFile);
			PrintError(status);
			result = FALSE;
			goto cleanup;
		}
		md5CalcStatus = apr_md5_update(&context, pFileBuffer, readBytes);
		if(md5CalcStatus != APR_SUCCESS) {
			PrintError(md5CalcStatus);
			result = FALSE;
			goto cleanup;
		}
	} while (status != APR_EOF);
	status = apr_md5_final(digest, &context);
	if (status != APR_SUCCESS) {
		PrintError(status);
	}
cleanup:
	if (pFileBuffer != NULL) {
		free(pFileBuffer);
		pFileBuffer = NULL;
	}
	status = apr_file_close(file);
	if (status != APR_SUCCESS) {
		PrintError(status);
	}
	return result;
}

int CalculateStringMd5(const char* pString, apr_byte_t* digest) {
	apr_status_t status = APR_SUCCESS;
	
	if (pString == NULL) {
		CrtPrintf("NULL string passed\n");
		return FALSE;
	}
	status = apr_md5(digest, pString, strlen(pString));
	if(status != APR_SUCCESS) {
		CrtPrintf("Failed to calculate MD5 of string: %s \n", pString);
		PrintError(status);
		return FALSE;
	}
	return TRUE;
}
