def load_fine_tuned_model(str_input):
    return str_input
def get_answer(obj_model, str_input):
    import time
    time.sleep(4)
    str_output='''
    <?xml version="1.0" encoding="UTF-8"?>
<ns0:PubOutages xmlns:ns0="http://iec.ch/TC57/2014/PubOutages#">
  <ns0:Outage>
    <!-- Unique outage event identifier -->
    <ns0:mRID>OUTAGE-COLUMBIANA-20250929</ns0:mRID>

    <!-- County-level descriptor (FIPS should be used if available) -->
    <ns0:communityDescriptor>COLUMBIANA</ns0:communityDescriptor>

    <!-- Outage metadata -->
    <ns0:cause>Pending Investigation</ns0:cause>
    <ns0:causeKind>unplanned</ns0:causeKind>
    <ns0:metersAffected>0</ns0:metersAffected>
    <ns0:outageKind>outageReported</ns0:outageKind>
    <ns0:statusKind>assigned</ns0:statusKind>
    <ns0:utilityDisclaimer>Preliminary data – subject to change</ns0:utilityDisclaimer>

    <!-- County outage area summary -->
    <ns0:OutageArea>
      <ns0:metersServed>31173</ns0:metersServed>
      <ns0:outageAreaKind>county</ns0:outageAreaKind>
    </ns0:OutageArea>

    <!-- Township/area details -->
    <ns0:OutageArea>
      <ns0:metersServed>1510</ns0:metersServed>
      <ns0:outageAreaKind>township</ns0:outageAreaKind>
      <ns0:communityDescriptor>FAIRFIELD TOWNSHIP</ns0:communityDescriptor>
    </ns0:OutageArea>

    <ns0:OutageArea>
      <ns0:metersServed>398</ns0:metersServed>
      <ns0:outageAreaKind>township</ns0:outageAreaKind>
      <ns0:communityDescriptor>HANOVERTON</ns0:communityDescriptor>
    </ns0:OutageArea>

    <ns0:OutageArea>
      <ns0:metersServed>65</ns0:metersServed>
      <ns0:outageAreaKind>township</ns0:outageAreaKind>
      <ns0:communityDescriptor>FRANKLIN TOWNSHIP</ns0:communityDescriptor>
    </ns0:OutageArea>

    <!-- Repeat ns0:OutageArea for each township in the input -->
    <!-- Example below for SALEM -->
    <ns0:OutageArea>
      <ns0:metersServed>6567</ns0:metersServed>
      <ns0:outageAreaKind>township</ns0:outageAreaKind>
      <ns0:communityDescriptor>SALEM</ns0:communityDescriptor>
    </ns0:OutageArea>

    <!-- Utility identifiers (EIA preferred) -->
    <ns0:Names>
      <ns0:name>ENTER_UTILITY_ID</ns0:name>
      <ns0:nameType>UtilityID</ns0:nameType>
      <ns0:nameTypeAuthority>EIA</ns0:nameTypeAuthority>
    </ns0:Names>
    <ns0:Names>
      <ns0:name>ENTER_UTILITY_NAME</ns0:name>
      <ns0:nameType>UtilityName</ns0:nameType>
      <ns0:nameTypeAuthority>EIA</ns0:nameTypeAuthority>
    </ns0:Names>
  </ns0:Outage>
</ns0:PubOutages>

    '''
    return str_output