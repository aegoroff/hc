/*
 * Copyright 2009 Alexander Egorov
 */

#include "targetver.h"

#include <stdio.h>
#include <math.h>
#include <locale.h>
#include "pglib.h"

#include "apr_pools.h"
#include "apr_getopt.h"
#include "apr_strings.h"
#include "apr_md5.h"
#include "apr_file_io.h"
#include "apr_mmap.h"
#include "apr_fnmatch.h"

#define BINARY_THOUSAND 1024
#define FILE_BIG_BUFFER_SIZE 1 * BINARY_THOUSAND * BINARY_THOUSAND // 1 megabyte
#define ERROR_BUFFER_SIZE 2 * BINARY_THOUSAND
#define BYTE_CHARS_SIZE 2 // byte representation string length
#define HEX_UPPER "%.2X"
#define HEX_LOWER "%.2x"
#define HLP_ARG "  -%c [ --%s ] arg\t\t%s\n"
#define HLP_NO_ARG "  -%c [ --%s ] \t\t%s\n"
#define MIN(x, y) ((x)<(y) ? (x):(y))

struct Version {
	WORD Major;
	WORD Minor;
	WORD Build;
	WORD Revision;
};

static struct apr_getopt_option_t options[] = {
	{ "file", 'f', TRUE, "input full file path to calculate MD5 sum for" },
	{ "dir", 'd', TRUE, "full path to dir to calculate MD5 of all content" },
	{ "exclude", 'e', TRUE, "exclude files that match the pattern specified" },
	{ "include", 'i', TRUE, "include only files that match the pattern specified" },
	{ "string", 's', TRUE, "string to calculate MD5 sum for" },
	{ "md5", 'm', TRUE, "MD5 hash to validate file or to find initial string (crack)" },
	{ "dict", 'a', TRUE, "initial string's dictionary by default all digits and upper and lower case latin symbols" },
	{ "crack", 'c', FALSE, "crack MD5 hash specified (find initial string) by option md5 (m)" },
	{ "lower", 'l', FALSE, "whether to output sum using low case" },
	{ "recursively", 'r', FALSE, "scan directory recursively" },
	{ "time", 't', FALSE, "show MD5 calculation time (false by default)" },
	{ "help", '?', FALSE, "show help message" }
};

static char* sizes[] = {
	"bytes",
	"Kb",
	"Mb",
	"Gb",
	"Tb",
	"Pb",
	"Eb",
	"Zb",
	"Yb"
};

static char* alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// Forward declarations
void PrintUsage(apr_pool_t* pool);
void PrintCopyright(apr_pool_t* pool);
int CalculateFileMd5(apr_pool_t* pool, const char* file, apr_byte_t* digest, int isPrintCalcTime);
void CalculateDirContentMd5(
						apr_pool_t* pool,
						const char* dir,
						int isPrintLowCase,
						int isScanDirRecursively,
						int isPrintCalcTime,
						const char* pExcludePattern,
						const char* pIncludePattern);
int CalculateStringMd5(const char* string, apr_byte_t* digest);
void PrintMd5(apr_byte_t* digest, int isPrintLowCase);
void CheckMd5(apr_byte_t* digest, const char* pCheckSum);
int CompareMd5(apr_byte_t* digest, const char* pCheckSum);
void PrintError(apr_status_t status);
void PrintSize(apr_off_t size);
void CrackMd5(apr_pool_t* pool, const char* pDict, const char* pCheckSum);
int MakeAttempt(apr_byte_t* digest, int* dictIndexes, int strSize, char* pStr, const char* pDict, unsigned long long* attemptsCount);
int CompareInputTo(apr_byte_t* digest, const void* input, int inputSize);
int CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2);
void ToDigest(const char* pCheckSum, apr_byte_t* digest);

/**
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* FromUtf8ToAnsi(const char* from, apr_pool_t* pool);
struct Version ReadVersion(apr_pool_t* pool, const char* pFile);

#ifdef WIN32
/**
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* DecodeUtf8Ansi(const char* from, apr_pool_t* pool, UINT fromCodePage, UINT toCodePage);
#endif

int main(int argc, const char * const argv[]) {
	apr_pool_t* pool = NULL;
	apr_getopt_t* opt = NULL;
	int c = 0;
	const char *optarg = NULL;
	const char *pFile = NULL;
	const char *pDir = NULL;
	const char *pCheckSum = NULL;
	const char *pString = NULL;
	const char *pExcludePattern = NULL;
	const char *pIncludePattern = NULL;
	const char *pDict = NULL;
	int isPrintLowCase = FALSE;
	int isScanDirRecursively = FALSE;
	int isPrintCalcTime = FALSE;
	int isCrack = FALSE;
	apr_byte_t digest[APR_MD5_DIGESTSIZE];
	apr_status_t status = APR_SUCCESS;

	setlocale(LC_ALL, ".ACP");
	setlocale(LC_NUMERIC, "C");

	status = apr_app_initialize(&argc, &argv, NULL);
	if (status != APR_SUCCESS) {
		CrtPrintf("Couldn't initialize APR\n");
		PrintError(status);
		return EXIT_FAILURE;
	}
	atexit(apr_terminate);
	apr_pool_create(&pool, NULL);
	apr_getopt_init(&opt, pool, argc, argv);

	if (argc < 2) {
		PrintUsage(pool);
		goto cleanup;
	}

	while ((status = apr_getopt_long(opt, options, &c, &optarg)) == APR_SUCCESS) {
		switch(c){
			case '?':
				PrintUsage(pool);
				goto cleanup;
			case 'f':
				pFile = apr_pstrdup(pool, optarg);
				break;
			case 'd':
				pDir = apr_pstrdup(pool, optarg);
				break;
			case 'm':
				pCheckSum = apr_pstrdup(pool, optarg);
				break;
			case 's':
				pString = apr_pstrdup(pool, optarg);
				break;
			case 'e':
				pExcludePattern = apr_pstrdup(pool, optarg);
				break;
			case 'i':
				pIncludePattern = apr_pstrdup(pool, optarg);
				break;
			case 'a':
				pDict = apr_pstrdup(pool, optarg);
				break;
			case 'l':
				isPrintLowCase = TRUE;
				break;
			case 'c':
				isCrack = TRUE;
				break;
			case 'r':
				isScanDirRecursively = TRUE;
				break;
			case 't':
				isPrintCalcTime = TRUE;
				break;
		}
	}

	if (status != APR_EOF) {
		PrintUsage(pool);
		goto cleanup;
	}
	if (pDict == NULL) {
		pDict = alphabet;
	}

	if (pFile != NULL && pCheckSum == NULL && !isCrack && CalculateFileMd5(pool, pFile, digest, isPrintCalcTime)) {
		PrintMd5(digest, isPrintLowCase);
	}
	if (pString != NULL && CalculateStringMd5(pString, digest)) {
		PrintMd5(digest, isPrintLowCase);
	}
	if (pCheckSum != NULL && pFile != NULL && CalculateFileMd5(pool, pFile, digest, isPrintCalcTime)) {
		CheckMd5(digest, pCheckSum);
	}
	if (pDir != NULL) {
		CalculateDirContentMd5(pool, pDir, isPrintLowCase, isScanDirRecursively, isPrintCalcTime, pExcludePattern, pIncludePattern);
	}
	if (pCheckSum != NULL && isCrack) {
		CrackMd5(pool, pDict, pCheckSum);
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

void PrintUsage(apr_pool_t* pool) {
	int i = 0;
	PrintCopyright(pool);
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

struct Version ReadVersion(apr_pool_t* pool, const char* pFile) {
	struct Version result = {0};
#ifdef WIN32
	DWORD sz = 0;
	UINT len = 0;
	VS_FIXEDFILEINFO* pFileInfo = NULL;
	BYTE* buffer = NULL;

	if (pFile == NULL) {
		return result;
	}

	sz = GetFileVersionInfoSizeA(pFile, NULL);

	if (sz) {
		buffer = (BYTE*)apr_pcalloc(pool, sz);

		if(!GetFileVersionInfoA(pFile, NULL, sz, buffer)) {
			return result;
		}

		if (!VerQueryValueA(buffer, "\\", (LPVOID*) &pFileInfo, &len)) {
			return result;
		}
		result.Major = HIWORD(pFileInfo->dwFileVersionMS);
		result.Minor = LOWORD(pFileInfo->dwFileVersionMS);
		result.Build = HIWORD(pFileInfo->dwFileVersionLS);
		result.Revision = LOWORD(pFileInfo->dwFileVersionLS);
	}
#endif
	return result;
}

void PrintCopyright(apr_pool_t* pool) {
	struct Version version = {0};
#ifdef WIN32
	char pApplicationExe[_MAX_PATH + 1];
	
	GetModuleFileNameA(NULL, pApplicationExe, _MAX_PATH);

	version = ReadVersion(pool, pApplicationExe);
#endif
	CrtPrintf(
		"\nMD5 Calculator %d.%d.%d.%d\nCopyright (C) 2009-2010 Alexander Egorov. All rights reserved.\n\n",
		version.Major,
		version.Minor,
		version.Build,
		version.Revision);
}

void PrintMd5(apr_byte_t* digest, int isPrintLowCase) {
	int i = 0;
	for (; i < APR_MD5_DIGESTSIZE; ++i) {
		CrtPrintf(isPrintLowCase ? HEX_LOWER : HEX_UPPER, digest[i]);
	}
	CrtPrintf("\n");
}

void CheckMd5(apr_byte_t* digest, const char* pCheckSum) {
	CrtPrintf("File is %s!\n", CompareMd5(digest, pCheckSum) ? "valid" : "invalid");
}

void ToDigest(const char* pCheckSum, apr_byte_t* digest) {
	int i = 0;
	for (; i < APR_MD5_DIGESTSIZE; ++i) {
		digest[i] = (apr_byte_t)htoi(pCheckSum + i * BYTE_CHARS_SIZE, BYTE_CHARS_SIZE);
	}
}

int CompareMd5(apr_byte_t* digest, const char* pCheckSum) {
	apr_byte_t bytes[APR_MD5_DIGESTSIZE];

	ToDigest(pCheckSum, bytes);
	return CompareDigests(bytes, digest);
}

void CrackMd5(apr_pool_t* pool, const char* pDict, const char* pCheckSum) {
	char* pStr = NULL;
	apr_byte_t digest[APR_MD5_DIGESTSIZE];
	int i = 0;
	int* dictIndexes = NULL;
	int* dictIndexesSubset = NULL;
	int dictSize = 0;
	int currentStrSize = 0;
	int isFound = FALSE;
	unsigned long long attemptsCount = 0;
	int ixDictIndexes = 0; // IMPORTANT: start from zero

#ifdef WIN32
		double span = 0;
		LARGE_INTEGER freq, time1, time2;
		
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&time1);
#endif

	// Empty string validation
	apr_md5(digest, NULL, 0);
	if (CompareMd5(digest, pCheckSum)) {
		pStr = "Empty string";
		isFound = TRUE;
		goto exit;
	}

	pStr = apr_pstrdup(pool, pDict);
	dictSize = strlen(pDict);

	ToDigest(pCheckSum, digest);

	dictIndexes = (int*)apr_pcalloc(pool, (dictSize + 1) * sizeof(int));
	if (dictIndexes == NULL) {
		CrtPrintf("Failed to allocate %i bytes for indexes array\n", (dictSize + 1) * sizeof(int));
		return;
	}
	dictIndexesSubset = (int*)apr_pcalloc(pool, (dictSize + 1) * sizeof(int));
	if (dictIndexesSubset == NULL) {
		CrtPrintf("Failed to allocate %i bytes for indexes subset array\n", (dictSize + 1) * sizeof(int));
		return;
	}

	while(!dictIndexesSubset[dictSize]) {
		i = 0;
		while(dictIndexesSubset[i]) { 
			dictIndexesSubset[i++] = 0;
		}
		dictIndexesSubset[i] = 1;
 		
		for(i = 0; i < dictSize; ++i) {
			if(dictIndexesSubset[i]) {
				dictIndexes[++ixDictIndexes] = i + 1; // IMPORTANT: Prefix increment
				++currentStrSize;
			}
		}

		if (MakeAttempt(digest, dictIndexes, currentStrSize + 1, pStr, pDict, &attemptsCount)) {
			isFound = TRUE;
			goto exit;
		}
		while (currentStrSize > 1 && !NextPermutation(currentStrSize, dictIndexes)) {
			if (MakeAttempt(digest, dictIndexes, currentStrSize + 1, pStr, pDict, &attemptsCount)) {
				isFound = TRUE;
				goto exit;
			}
		}
		currentStrSize = 0;
		ixDictIndexes = 0; // IMPORTANT: start from zero
	}
	
exit:
#ifdef WIN32
	QueryPerformanceCounter(&time2);
	span = (double) (time2.QuadPart - time1.QuadPart) / (double)freq.QuadPart;
	CrtPrintf("\nAttempts: %llu Time %.3f sec\n", attemptsCount, span);
#endif
	if (isFound) {
		CrtPrintf("Initial string is: %s \n", pStr);
	} else {
		CrtPrintf("Nothing found\n");
	}
}

int MakeAttempt(apr_byte_t* digest, int* dictIndexes, int strSize, char* pStr, const char* pDict, unsigned long long* attemptsCount) {
	int i = 1;

	++*attemptsCount;
	for (; i < strSize; ++i) {
		pStr[i - 1] = pDict[dictIndexes[i] - 1];
	}
	pStr[strSize - 1] = 0;

	return CompareInputTo(digest, pStr, strSize - 1);
}

int CompareInputTo(apr_byte_t* digest, const void* input, int inputSize) {
	apr_byte_t digestAttempt[APR_MD5_DIGESTSIZE];
	
	apr_md5(digestAttempt, input, inputSize);
	return CompareDigests(digestAttempt, digest);
}

/**
* It's so ugly to improve performance (to be inlined)
* Don't bother about multiplication here - compiler optimizes it
*/
int CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2) {
	if ((apr_uint32_t)digest1[0] != (apr_uint32_t)digest2[0]) {
		return FALSE;
	}
	if ((apr_uint32_t)digest1[sizeof(apr_uint32_t)] != (apr_uint32_t)digest2[sizeof(apr_uint32_t)]) {
		return FALSE;
	}
	if ((apr_uint32_t)digest1[sizeof(apr_uint32_t) * 2] != (apr_uint32_t)digest2[sizeof(apr_uint32_t) * 2]) {
		return FALSE;
	}
	if ((apr_uint32_t)digest1[sizeof(apr_uint32_t) * 3] != (apr_uint32_t)digest2[sizeof(apr_uint32_t) * 3]) {
		return FALSE;
	}
	return TRUE;
}

void CalculateDirContentMd5(
							apr_pool_t* pool,
							const char* dir,
							int isPrintLowCase,
							int isScanDirRecursively,
							int isPrintCalcTime,
							const char* pExcludePattern,
							const char* pIncludePattern) {
	apr_finfo_t info;
	apr_dir_t* d = NULL;
	apr_status_t status = APR_SUCCESS;
	apr_byte_t digest[APR_MD5_DIGESTSIZE];
	char* fullPathToFile = NULL;
	apr_pool_t* filePool = NULL;
	apr_pool_t* dirPool = NULL;

	apr_pool_create(&filePool, pool);
	apr_pool_create(&dirPool, pool);

	status = apr_dir_open(&d, dir, dirPool);
	if (status != APR_SUCCESS) {
		PrintError(status);
		return;
	}
	
	for (;;) {
		apr_pool_clear(filePool); // cleanup file allocated memory
		status = apr_dir_read(&info, APR_FINFO_NAME | APR_FINFO_MIN, d);
		if (APR_STATUS_IS_ENOENT(status)) {
            break;
        }
		if (info.filetype == APR_DIR && isScanDirRecursively) {
			if ((info.name[0] == '.' && info.name[1] == '\0')
				|| (info.name[0] == '.' && info.name[1] == '.' && info.name[2] == '\0')) {
				continue;
			}

			status = apr_filepath_merge(&fullPathToFile, dir, info.name, APR_FILEPATH_NATIVE, filePool);
			if (status != APR_SUCCESS) {
				PrintError(status);
				goto cleanup;
			}
			CalculateDirContentMd5(pool, fullPathToFile, isPrintLowCase, isScanDirRecursively, isPrintCalcTime, pExcludePattern, pIncludePattern);
        }
        if (status != APR_SUCCESS || info.filetype != APR_REG) {
            continue;
        }

		if (pIncludePattern && apr_fnmatch(pIncludePattern, info.name, APR_FNM_CASE_BLIND) == APR_FNM_NOMATCH) {
			continue;
		}
		if (pExcludePattern && apr_fnmatch(pExcludePattern, info.name, APR_FNM_CASE_BLIND) == APR_SUCCESS) {
			continue;
		}

		status = apr_filepath_merge(&fullPathToFile, dir, info.name, APR_FILEPATH_NATIVE, filePool);
		if (status != APR_SUCCESS) {
			PrintError(status);
			goto cleanup;
		}

		if (CalculateFileMd5(filePool, fullPathToFile, digest, isPrintCalcTime)) {
			PrintMd5(digest, isPrintLowCase);
		}
	}

cleanup:
	apr_pool_destroy(dirPool);
	apr_pool_destroy(filePool);
	status = apr_dir_close(d);
	if (status != APR_SUCCESS) {
		PrintError(status);
	}
}

int CalculateFileMd5(apr_pool_t* pool, const char* pFile, apr_byte_t* digest, int isPrintCalcTime) {
	apr_file_t* file = NULL;
	apr_finfo_t info;
	apr_md5_ctx_t context = {0};
	apr_status_t status = APR_SUCCESS;
	apr_status_t md5CalcStatus = APR_SUCCESS;
	int result = TRUE;
	apr_off_t strSize = 0;
	apr_mmap_t* mmap = NULL;
	apr_off_t offset = 0;
	char* pFileAnsi = NULL;
	
	#ifdef WIN32
		double span = 0;
		LARGE_INTEGER freq, time1, time2;
	#endif

	pFileAnsi = FromUtf8ToAnsi(pFile, pool);
	CrtPrintf("%s | ", pFileAnsi == NULL ? pFile : pFileAnsi);

#ifdef WIN32
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&time1);
#endif

	status = apr_file_open(&file, pFile, APR_READ | APR_BINARY, APR_FPROT_WREAD, pool);
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

	status = apr_file_info_get(&info, APR_FINFO_NAME | APR_FINFO_MIN, file);

	if(status != APR_SUCCESS) {
		PrintError(status);
		result = FALSE;
		goto cleanup;
	}

	PrintSize(info.size);
	CrtPrintf(" | ");

	if (info.size > FILE_BIG_BUFFER_SIZE) {
		strSize = FILE_BIG_BUFFER_SIZE;
	} else if (info.size == 0) {
		status = apr_md5(digest, NULL, 0);
		goto endtiming;
	} else {
		strSize = info.size;
	}

	do {
		status = apr_mmap_create(&mmap, file, offset, MIN(strSize, info.size - offset), APR_MMAP_READ, pool);
		if(status != APR_SUCCESS) {
			PrintError(status);
			result = FALSE;
			mmap = NULL;
			goto cleanup;
		}
		md5CalcStatus = apr_md5_update(&context, mmap ->mm, mmap ->size);
		if(md5CalcStatus != APR_SUCCESS) {
			PrintError(md5CalcStatus);
			result = FALSE;
			goto cleanup;
		}
		offset += mmap ->size;
		status = apr_mmap_delete(mmap);
		if(status != APR_SUCCESS) {
			PrintError(status);
			mmap = NULL;
			result = FALSE;
			goto cleanup;
		}
		mmap = NULL;
	} while (offset < info.size);
	status = apr_md5_final(digest, &context);
endtiming:
#ifdef WIN32
	QueryPerformanceCounter(&time2);
	span = (double) (time2.QuadPart - time1.QuadPart) / (double)freq.QuadPart;
	if (isPrintCalcTime) {
		CrtPrintf("%.3f sec | ", span);
	}
#endif
	if (status != APR_SUCCESS) {
		PrintError(status);
	}
cleanup:
	if (mmap != NULL) {
		status = apr_mmap_delete(mmap);
		mmap = NULL;
		if (status != APR_SUCCESS) {
			PrintError(status);
		}
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

void PrintSize(apr_off_t size) {
	int expr = 0;
	expr = size == 0 ? 0 : floor(log(size)/log(BINARY_THOUSAND));
	if (expr == 0) {
		CrtPrintf("%lld %s", size, sizes[expr]);
	} else {
		CrtPrintf("%.2f %s", size / pow(BINARY_THOUSAND, floor(expr)), sizes[expr]);
	}
}

/**
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* FromUtf8ToAnsi(const char* from, apr_pool_t* pool) {
#ifdef WIN32
	return DecodeUtf8Ansi(from, pool, CP_UTF8, CP_ACP);
#else
	return NULL;
#endif
}

#ifdef WIN32
/**
* IMPORTANT: Memory allocated for result must be freed up by caller
*/
char* DecodeUtf8Ansi(const char* from, apr_pool_t* pool, UINT fromCodePage, UINT toCodePage) {
	int lengthWide = 0;
	int lengthAnsi = 0;
	size_t cbFrom = 0;
	wchar_t* wideStr = NULL;
	char* ansiStr = NULL;
	apr_size_t wideBufferSize = 0;

	cbFrom = strlen(from) + 1; // IMPORTANT!!! including null terminator

	lengthWide = MultiByteToWideChar(fromCodePage, 0, from, cbFrom, NULL, 0); // including null terminator
	wideBufferSize = sizeof(wchar_t) * lengthWide;
	wideStr = pool == NULL ? (wchar_t*)malloc(wideBufferSize) : (wchar_t*)apr_pcalloc(pool, wideBufferSize);
	if (wideStr == NULL) {
		CrtPrintf("Failed to allocate %i bytes\n", wideBufferSize);
		return NULL;
	}
	MultiByteToWideChar(fromCodePage, 0, from, cbFrom, wideStr, lengthWide);
	
	lengthAnsi = WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, 0, NULL, NULL); // null terminator included
	ansiStr = (char*)apr_pcalloc(pool, lengthAnsi);

	if (ansiStr == NULL) {
		CrtPrintf("Failed to allocate %i bytes\n", lengthAnsi);
		goto cleanup;
	}
	WideCharToMultiByte(toCodePage, 0, wideStr, lengthWide, ansiStr, lengthAnsi, NULL, NULL);

cleanup:
	if (pool == NULL) { // allocation wasn't from Apache pool
		free(wideStr);
	}

	return ansiStr;
}
#endif