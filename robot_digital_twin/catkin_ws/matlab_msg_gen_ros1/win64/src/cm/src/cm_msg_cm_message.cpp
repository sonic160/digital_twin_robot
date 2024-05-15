// Copyright 2019-2021 The MathWorks, Inc.
// Common copy functions for cm/msg_cm
#include "boost/date_time.hpp"
#include "boost/shared_array.hpp"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4265)
#pragma warning(disable : 4458)
#pragma warning(disable : 4100)
#pragma warning(disable : 4127)
#pragma warning(disable : 4267)
#pragma warning(disable : 4068)
#pragma warning(disable : 4245)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif //_MSC_VER
#include "ros/ros.h"
#include "cm/msg_cm.h"
#include "visibility_control.h"
#include "MATLABROSMsgInterface.hpp"
#include "ROSPubSubTemplates.hpp"
class CM_EXPORT cm_msg_msg_cm_common : public MATLABROSMsgInterface<cm::msg_cm> {
  public:
    virtual ~cm_msg_msg_cm_common(){}
    virtual void copy_from_struct(cm::msg_cm* msg, const matlab::data::Struct& arr, MultiLibLoader loader); 
    //----------------------------------------------------------------------------
    virtual MDArray_T get_arr(MDFactory_T& factory, const cm::msg_cm* msg, MultiLibLoader loader, size_t size = 1);
};
  void cm_msg_msg_cm_common::copy_from_struct(cm::msg_cm* msg, const matlab::data::Struct& arr,
               MultiLibLoader loader) {
    try {
        //header
        const matlab::data::StructArray header_arr = arr["Header"];
        auto msgClassPtr_header = getCommonObject<std_msgs::Header>("std_msgs_msg_Header_common",loader);
        msgClassPtr_header->copy_from_struct(&msg->header,header_arr[0],loader);
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Header' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Header' is wrong type; expected a struct.");
    }
    try {
        //name
        const matlab::data::CellArray name_cellarr = arr["Name"];
        size_t nelem = name_cellarr.getNumberOfElements();
        for (size_t idx=0; idx < nelem; ++idx){
        	const matlab::data::CharArray name_arr = name_cellarr[idx];
        	msg->name.push_back(name_arr.toAscii());
        }
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Name' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Name' is wrong type; expected a string.");
    }
    try {
        //position
        const matlab::data::TypedArray<double> position_arr = arr["Position"];
        size_t nelem = position_arr.getNumberOfElements();
        	msg->position.resize(nelem);
        	std::copy(position_arr.begin(), position_arr.begin()+nelem, msg->position.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Position' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Position' is wrong type; expected a double.");
    }
    try {
        //temperature
        const matlab::data::TypedArray<double> temperature_arr = arr["Temperature"];
        size_t nelem = temperature_arr.getNumberOfElements();
        	msg->temperature.resize(nelem);
        	std::copy(temperature_arr.begin(), temperature_arr.begin()+nelem, msg->temperature.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Temperature' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Temperature' is wrong type; expected a double.");
    }
    try {
        //voltage
        const matlab::data::TypedArray<double> voltage_arr = arr["Voltage"];
        size_t nelem = voltage_arr.getNumberOfElements();
        	msg->voltage.resize(nelem);
        	std::copy(voltage_arr.begin(), voltage_arr.begin()+nelem, msg->voltage.begin());
    } catch (matlab::data::InvalidFieldNameException&) {
        throw std::invalid_argument("Field 'Voltage' is missing.");
    } catch (matlab::Exception&) {
        throw std::invalid_argument("Field 'Voltage' is wrong type; expected a double.");
    }
  }
  //----------------------------------------------------------------------------
  MDArray_T cm_msg_msg_cm_common::get_arr(MDFactory_T& factory, const cm::msg_cm* msg,
       MultiLibLoader loader, size_t size) {
    auto outArray = factory.createStructArray({size,1},{"MessageType","Header","Name","Position","Temperature","Voltage"});
    for(size_t ctr = 0; ctr < size; ctr++){
    outArray[ctr]["MessageType"] = factory.createCharArray("cm/msg_cm");
    // header
    auto currentElement_header = (msg + ctr)->header;
    auto msgClassPtr_header = getCommonObject<std_msgs::Header>("std_msgs_msg_Header_common",loader);
    outArray[ctr]["Header"] = msgClassPtr_header->get_arr(factory, &currentElement_header, loader);
    // name
    auto currentElement_name = (msg + ctr)->name;
    auto nameoutCell = factory.createCellArray({currentElement_name.size(),1});
    for(size_t idxin = 0; idxin < currentElement_name.size(); ++ idxin){
    	nameoutCell[idxin] = factory.createCharArray(currentElement_name[idxin]);
    }
    outArray[ctr]["Name"] = nameoutCell;
    // position
    auto currentElement_position = (msg + ctr)->position;
    outArray[ctr]["Position"] = factory.createArray<cm::msg_cm::_position_type::const_iterator, double>({currentElement_position.size(),1}, currentElement_position.begin(), currentElement_position.end());
    // temperature
    auto currentElement_temperature = (msg + ctr)->temperature;
    outArray[ctr]["Temperature"] = factory.createArray<cm::msg_cm::_temperature_type::const_iterator, double>({currentElement_temperature.size(),1}, currentElement_temperature.begin(), currentElement_temperature.end());
    // voltage
    auto currentElement_voltage = (msg + ctr)->voltage;
    outArray[ctr]["Voltage"] = factory.createArray<cm::msg_cm::_voltage_type::const_iterator, double>({currentElement_voltage.size(),1}, currentElement_voltage.begin(), currentElement_voltage.end());
    }
    return std::move(outArray);
  } 
class CM_EXPORT cm_msg_cm_message : public ROSMsgElementInterfaceFactory {
  public:
    virtual ~cm_msg_cm_message(){}
    virtual std::shared_ptr<MATLABPublisherInterface> generatePublisherInterface(ElementType type);
    virtual std::shared_ptr<MATLABSubscriberInterface> generateSubscriberInterface(ElementType type);
    virtual std::shared_ptr<MATLABRosbagWriterInterface> generateRosbagWriterInterface(ElementType type);
};  
  std::shared_ptr<MATLABPublisherInterface> 
          cm_msg_cm_message::generatePublisherInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSPublisherImpl<cm::msg_cm,cm_msg_msg_cm_common>>();
  }
  std::shared_ptr<MATLABSubscriberInterface> 
         cm_msg_cm_message::generateSubscriberInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSSubscriberImpl<cm::msg_cm,cm::msg_cm::ConstPtr,cm_msg_msg_cm_common>>();
  }
#include "ROSbagTemplates.hpp" 
  std::shared_ptr<MATLABRosbagWriterInterface>
         cm_msg_cm_message::generateRosbagWriterInterface(ElementType type){
    if(type != eMessage){
        throw std::invalid_argument("Wrong input, Expected eMessage");
    }
    return std::make_shared<ROSBagWriterImpl<cm::msg_cm,cm_msg_msg_cm_common>>();
  }
#include "register_macro.hpp"
// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
CLASS_LOADER_REGISTER_CLASS(cm_msg_msg_cm_common, MATLABROSMsgInterface<cm::msg_cm>)
CLASS_LOADER_REGISTER_CLASS(cm_msg_cm_message, ROSMsgElementInterfaceFactory)
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif //_MSC_VER
//gen-1