tree grammar HLINQWalker;

options {
	tokenVocab	    = HLINQ;
    ASTLabelType    = pANTLR3_BASE_TREE;
    language	    = C;
    output=AST;
}

@header {
   #include "compiler.h"
#ifdef GTEST
  #include "displayError.h"
#endif
}
 
@apifuncs {
#ifdef GTEST
  RECOGNIZER->displayRecognitionError       = displayRecognitionErrorNew;
#endif
}

@members {
	BOOL printCalcTime;
}

prog[apr_pool_t* root, BOOL onlyValidate, BOOL isPrintCalcTime]
@init { 
	printCalcTime = $isPrintCalcTime;
	InitProgram($onlyValidate, $root); 
}
	: statement*
	;
     
statement
@init {
        OpenStatement(); 
}
@after {
        CloseStatement(printCalcTime);
}
    :   expr
    ;

expr:
	FOR 
	( { DefineQueryType(CtxTypeString); } expr_string 
	| { DefineQueryType(CtxTypeHash); } expr_hash  
	| { DefineQueryType(CtxTypeDir); } expr_dir
	| { DefineQueryType(CtxTypeFile); } expr_file 
	)
    ;

expr_string:
	^(HASH_STR {  RegisterIdentifier("_s_"); } hash_clause source)
	;

expr_hash:
	^(BRUTE_FORCE id brute_force_clause let_clause? source)
	;

expr_dir
	: ^(HASH_DIR hash_clause id let_clause? where_clause? (WITHSUBS { SetRecursively(); })? source)
	| ^(HASH_DIR id let_clause? where_clause? FIND (WITHSUBS { SetRecursively(); })? source)
	;

expr_file
	: ^(HASH_FILE hash_clause id let_clause? source)
	;
	
source : s=STRING { SetSource($s.text->chars); };
    
id : ID { RegisterIdentifier($ID.text->chars); };

attr_clause : ^(ATTR_REF ID attr) ;

attr : str_attr | int_attr ;

hash_clause
    : MD5 {  SetHashAlgorithm(AlgMd5); }
    | MD4 {  SetHashAlgorithm(AlgMd4); }
    | SHA1 {  SetHashAlgorithm(AlgSha1); }
    | SHA256 {  SetHashAlgorithm(AlgSha256); }
    | SHA384 {  SetHashAlgorithm(AlgSha384); }
    | SHA512 {  SetHashAlgorithm(AlgSha512); }
    | CRC32 {  SetHashAlgorithm(AlgCrc32); }
    | WHIRLPOOL {  SetHashAlgorithm(AlgWhirlpool); }
    ;
    
brute_force_clause
	: CRACK hash_clause { SetBruteForce(); }
	;

let_clause
	: assign+
	;

where_clause
	: boolean_expression
    ;

boolean_expression
	: ^(EQUAL boolean_expression boolean_expression)
	| ^(NOTEQUAL boolean_expression boolean_expression)
	| ^(MATCH boolean_expression boolean_expression)
	| ^(NOTMATCH boolean_expression boolean_expression)
	| ^(LE boolean_expression boolean_expression)
	| ^(GE boolean_expression boolean_expression)
	| ^(LEASSIGN boolean_expression boolean_expression)
	| ^(GEASSIGN boolean_expression boolean_expression)
	| ^(OR boolean_expression boolean_expression)
	| ^(AND boolean_expression boolean_expression)
	| ^(ATTR_REF ID boolean_expression)
	| STRING
	| INT
	| ID
	| str_attr
	| int_attr
	;

assign 
	:	^(ATTR_REF ID ^(ASSIGN_OP str_attr STRING))
	|	^(ATTR_REF ID ^(ASSIGN_OP int_attr INT))
	;
 
str_attr : NAME_ATTR | PATH_ATTR | DICT_ATTR | hash_clause ; 

int_attr : SIZE_ATTR | LIMIT_ATTR | OFFSET_ATTR | MIN_ATTR | MAX_ATTR ; 

