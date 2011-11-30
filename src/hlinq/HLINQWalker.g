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

boolean_expression returns [pANTLR3_UINT8 value, StrAttr strCode, IntAttr intCode]
@init { 
	$value = NULL;
	$strCode = StrAttrUndefined;
	$intCode = IntAttrUndefined;
}
	: ^(EQUAL l=boolean_expression r=boolean_expression) { WhereClauseCall($l.intCode, $l.strCode, $r.value, CondOpEq); }
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
	| STRING { $value = $STRING.text->chars; }
	| INT { $value = $INT.text->chars; }
	| NAME_ATTR { $strCode = StrAttrName; }
	| PATH_ATTR { $strCode = StrAttrPath; }
	| MD5 { $strCode = StrAttrMd5; }
	| MD4 { $strCode = StrAttrMd4; }
	| SHA1 { $strCode = StrAttrSha1; }
	| SHA256 { $strCode = StrAttrSha256; }
	| SHA384 { $strCode = StrAttrSha384; }
	| SHA512 { $strCode = StrAttrSha512; }
	| CRC32 { $strCode = StrAttrCrc32; }
	| WHIRLPOOL { $strCode = StrAttrWhirlpool; }
	| SIZE_ATTR { $intCode = IntAttrSize; }
	| LIMIT_ATTR
	| OFFSET_ATTR
	;

assign 
	:	^(ATTR_REF ID ^(ASSIGN_OP sa=str_attr s=STRING)) { AssignStrAttribute($sa.code, $s.text->chars); }
	|	^(ATTR_REF ID ^(ASSIGN_OP ia=int_attr i=INT)) { AssignIntAttribute($ia.code, $i.text->toInt32($i.text)); }
	;
 
str_attr returns[StrAttr code] 
@init { $code = StrAttrUndefined; }
	: NAME_ATTR  { $code = StrAttrName; }
	| PATH_ATTR  { $code = StrAttrPath; }
	| DICT_ATTR  { $code = StrAttrDict; }
	| MD5 { $code = StrAttrMd5; }
	| MD4 { $code = StrAttrMd4; }
	| SHA1 { $code = StrAttrSha1; }
	| SHA256 { $code = StrAttrSha256; }
	| SHA384 { $code = StrAttrSha384; }
	| SHA512 { $code = StrAttrSha512; }
	| CRC32 { $code = StrAttrCrc32; }
	| WHIRLPOOL { $code = StrAttrWhirlpool; }; 

int_attr returns[IntAttr code]
@init { $code = IntAttrUndefined; }
	: SIZE_ATTR { $code = IntAttrSize; } 
	| LIMIT_ATTR { $code = IntAttrLimit; } 
	| OFFSET_ATTR { $code = IntAttrOffset; } 
	| MIN_ATTR { $code = IntAttrMin; } 
	| MAX_ATTR { $code = IntAttrMax; } ; 

