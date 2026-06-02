(define (problem picture_58)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_sensor blue_pump red_sensor - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_sensor)
        (clear blue_pump)
        (part_at blue_sensor table)
        (part_at blue_pump table)
        (part_at red_sensor table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
        )
    )
)
