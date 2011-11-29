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
	: statement+ | EOF
	;

     
statement
@init {
	OpenStatement(); 
}
@after {
	CloseStatement(RECOGNIZER->state->errorCount, printCalcTime);
}
    :   expr NEWLINE | NEWLINE
    ;

expr:
	FOR (expr_string | expr_hash | expr_dir | expr_file)
    ;

expr_string:
	STR {  RegisterIdentifier("_s_", CtxTypeString); } s=STRING { SetSource($s.text->chars); } DO hash_clause
	;

expr_hash:
	STR id[CtxTypeHash] FROM HASH s=STRING { SetSource($s.text->chars); } (let_clause)? DO brute_force_clause
	;

expr_dir:
	FILE id[CtxTypeDir] FROM DIR s=STRING { SetSource($s.text->chars); } (let_clause)? (where_clause)? DO (hash_clause | find_clause) (recursively)?
	;

expr_file:
	FILE id[CtxTypeFile] FROM s=STRING { SetSource($s.text->chars); } (let_clause)? DO hash_clause
	;
    
id[CtxType contextType]
	: ID
	{
		 RegisterIdentifier($ID.text->chars, $contextType);
	};

attr_clause: ID DOT attr ;

attr:
    ( str_attr | int_attr )
    ;

find_clause:
    'find'
    ;

hash_clause:
    (md5 | md4 | sha1 | sha256 | sha384 | sha512 | crc32 | whirlpool)
    ;
  
md5	:	MD5 {  SetHashAlgorithm(AlgMd5); };
md4	:	MD4 {  SetHashAlgorithm(AlgMd4); };
sha1	:	SHA1 {  SetHashAlgorithm(AlgSha1); };
sha256	:	SHA256 {  SetHashAlgorithm(AlgSha256); };
sha384	:	SHA384 {  SetHashAlgorithm(AlgSha384); };
sha512	:	SHA512 {  SetHashAlgorithm(AlgSha512); };
crc32	:	CRC32 {  SetHashAlgorithm(AlgCrc32); };
whirlpool	:	WHIRLPOOL {  SetHashAlgorithm(AlgWhirlpool); };
    
brute_force_clause
	:	'crack' hash_clause 
	{ 
		SetBruteForce();
	}
	;

recursively
	: 'recursively'	
	{
		SetRecursively();
	}
	;

let_clause:
	LET assign (COMMA assign)*
	;

where_clause:
    'where' boolean_expression
    ;

boolean_expression:
	conditional_or_expression;

conditional_or_expression:
	conditional_and_expression  (OR^ conditional_and_expression)* ;

conditional_and_expression:
	exclusive_or_expression   (AND^ exclusive_or_expression)* ;

exclusive_or_expression:
	(relational_expr_str | relational_expr_int)
	|
	OPEN_BRACE! boolean_expression CLOSE_BRACE!
	;

relational_expr_str
	:	ID DOT^ (str_attr EQUAL^ STRING | str_attr NOTEQUAL^ STRING | str_attr MATCH^ STRING | str_attr NOTMATCH^ STRING)
	;

relational_expr_int
	:	ID DOT^ (int_attr EQUAL^ INT | int_attr NOTEQUAL^ INT | int_attr GE^ INT | int_attr LE^ INT | int_attr LEASSIGN^ INT | int_attr GEASSIGN^ INT)
	;

assign :
	ID DOT^ ((str_attr ASSIGN_OP^ STRING) | (int_attr ASSIGN_OP^ INT))
	;
 
str_attr returns[StrAttr code]
@init { $code = StrAttrUndefined; }
:
    (
    'name' { $code = StrAttrName; } | 
    'path' { $code = StrAttrPath; } | 
    'dict' { $code = StrAttrDict; } | 
     MD5 { $code = StrAttrMd5; } | 
     SHA1 { $code = StrAttrSha1; } | 
     SHA256 { $code = StrAttrSha256; } | 
     SHA384 { $code = StrAttrSha384; } | 
     SHA512 { $code = StrAttrSha512; } | 
     MD4 { $code = StrAttrMd4; } | 
     CRC32 { $code = StrAttrCrc32; } | 
     WHIRLPOOL { $code = StrAttrWhirlpool; } 
     )
    ; 

int_attr returns[IntAttr code]
@init { $code = IntAttrUndefined; }
:
    ('size' { $code = IntAttrSize; } | 'limit' { $code = IntAttrLimit; } | 'offset' { $code = IntAttrOffset; } | 'min' { $code = IntAttrMin; } | 'max' { $code = IntAttrMax; } )
    ; 

