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
    :   expr
    ;

expr:
	FOR (expr_string | expr_hash | expr_dir | expr_file)
    ;

expr_string:
	^(HASH_STR hash_clause STRING)
	;

expr_hash:
	^(BRUTE_FORCE brute_force_clause id let_clause? STRING)
	;

expr_dir
	: ^(HASH_DIR hash_clause id let_clause? where_clause? WITHSUBS? STRING)
	| ^(HASH_DIR id let_clause? where_clause? FIND WITHSUBS? STRING)
	;

expr_file
	: ^(HASH_FILE hash_clause id let_clause? STRING)
	;
    
id : ID;

attr_clause : ^(ATTR_REF ID attr) ;

attr : str_attr | int_attr ;

hash_clause
    : MD5 | MD4 | SHA1 | SHA256 | SHA384 | SHA512 | CRC32 | WHIRLPOOL
    ;
    
brute_force_clause
	: CRACK hash_clause 
	;

let_clause
	: assign+
	;

where_clause
	: boolean_expression
    ;

boolean_expression
	: ^(EQUAL str_attr STRING)
	| ^(NOTEQUAL str_attr STRING)
	| ^(MATCH str_attr STRING)
	| ^(NOTMATCH str_attr STRING)
	| ^(EQUAL int_attr INT)
	| ^(NOTEQUAL int_attr INT)
	| ^(LE int_attr INT)
	| ^(GE int_attr INT)
	| ^(LEASSIGN int_attr INT)
	| ^(GEASSIGN int_attr INT)
	| ^(OR boolean_expression boolean_expression)
	| ^(AND boolean_expression boolean_expression)
	;

assign 
	:	^(ATTR_REF ID ^(ASSIGN_OP str_attr STRING))
	|	^(ATTR_REF ID ^(ASSIGN_OP int_attr INT))
	;
 
str_attr : NAME_ATTR | PATH_ATTR | DICT_ATTR | hash_clause ; 

int_attr : SIZE_ATTR | LIMIT_ATTR | OFFSET_ATTR | MIN_ATTR | MAX_ATTR ; 

