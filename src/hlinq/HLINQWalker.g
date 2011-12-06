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
	BOOL printLowCase;
}

prog[apr_pool_t* root, BOOL onlyValidate, BOOL isPrintCalcTime, BOOL isPrintLowCase]
@init { 
	printCalcTime = $isPrintCalcTime;
	printLowCase = $isPrintLowCase;
	InitProgram($onlyValidate, $root); 
}
	: statement*
	;
     
statement
@init {
        OpenStatement(RECOGNIZER->state); 
}
@after {
        CloseStatement(printCalcTime, printLowCase);
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
	| ^(HASH_FILE id let_clause? source)
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

boolean_expression returns [pANTLR3_UINT8 value, Attr code]
@init { 
	$value = NULL;
	$code = AttrUndefined;
}
	: ^(EQUAL l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpEq); }
	| ^(NOTEQUAL l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpNotEq); }
	| ^(MATCH l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpMatch); }
	| ^(NOTMATCH l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpNotMatch); }
	| ^(LE l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpLe); }
	| ^(GE l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpGe); }
	| ^(LEASSIGN l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpLeEq); }
	| ^(GEASSIGN l=boolean_expression r=boolean_expression) { WhereClauseCall($l.code, $r.value, CondOpGeEq); }
	| ^(OR boolean_expression boolean_expression) { WhereClauseCond(CondOpOr); }
	| ^(AND boolean_expression boolean_expression) { WhereClauseCond(CondOpAnd); }
	| ^(NOT_OP boolean_expression) { WhereClauseCond(CondOpNot); }
	| ^(ATTR_REF ID boolean_expression) { CallAttiribute($ID.text->chars, $ID); }
	| STRING { $value = $STRING.text->chars; }
	| INT { $value = $INT.text->chars; }
	| NAME_ATTR { $code = AttrName; }
	| PATH_ATTR { $code = AttrPath; }
	| MD5 { $code = AttrMd5; }
	| MD4 { $code = AttrMd4; }
	| SHA1 { $code = AttrSha1; }
	| SHA256 { $code = AttrSha256; }
	| SHA384 { $code = AttrSha384; }
	| SHA512 { $code = AttrSha512; }
	| CRC32 { $code = AttrCrc32; }
	| WHIRLPOOL { $code = AttrWhirlpool; }
	| SIZE_ATTR { $code = AttrSize; }
	| LIMIT_ATTR { $code = AttrLimit; }
	| OFFSET_ATTR { $code = AttrOffset; }
	;

assign 
	:	^(ATTR_REF ID ^(ASSIGN_OP sa=str_attr s=STRING)) 
	{ 
		if(CallAttiribute($ID.text->chars, $ID)) {
			AssignAttribute($sa.code, $s.text->chars);
		}
	}
	|	^(ATTR_REF ID ^(ASSIGN_OP ia=int_attr i=INT))
	{ 
		if(CallAttiribute($ID.text->chars, $ID)){
			AssignAttribute($ia.code, $i.text->chars);
		}
	}
	;
 
str_attr returns[Attr code] 
@init { $code = AttrUndefined; }
	: NAME_ATTR  { $code = AttrName; }
	| PATH_ATTR  { $code = AttrPath; }
	| DICT_ATTR  { $code = AttrDict; }
	| MD5 { $code = AttrMd5; }
	| MD4 { $code = AttrMd4; }
	| SHA1 { $code = AttrSha1; }
	| SHA256 { $code = AttrSha256; }
	| SHA384 { $code = AttrSha384; }
	| SHA512 { $code = AttrSha512; }
	| CRC32 { $code = AttrCrc32; }
	| WHIRLPOOL { $code = AttrWhirlpool; }; 

int_attr returns[Attr code]
@init { $code = AttrUndefined; }
	: SIZE_ATTR { $code = AttrSize; } 
	| LIMIT_ATTR { $code = AttrLimit; } 
	| OFFSET_ATTR { $code = AttrOffset; } 
	| MIN_ATTR { $code = AttrMin; } 
	| MAX_ATTR { $code = AttrMax; } ; 

